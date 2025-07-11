# tvr/models/modeling_qlora.py

import torch
from torch import nn
from transformers import CLIPModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

from .until_module import AllGather, CrossEn # Import các module cần thiết từ dự án

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class TempMeQLoRA(nn.Module):
    """
    Lớp này tích hợp QLoRA vào kiến trúc TempMe.
    Giai đoạn 2: Tích hợp cơ bản, tạm thời vô hiệu hóa gộp token.
    """
    def __init__(self, config): # Nhận toàn bộ config (args)
        super().__init__()
        self.config = config
        
        # 1. Xác định tên mô hình từ config
        backbone_name = "openai/clip-vit-base-patch32"
        if config.base_encoder == "ViT-B/16":
            backbone_name = "openai/clip-vit-base-patch16"
        logger.info(f"Đang tải backbone QLoRA: {backbone_name}")

        # 2. Chuẩn bị cấu hình QLoRA
        qlora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 3. Tải mô hình CLIP từ Transformers
        model = CLIPModel.from_pretrained(
            backbone_name,
            quantization_config=qlora_config,
            device_map="auto",
        )
        
        # 4. Chuẩn bị và áp dụng PEFT/LoRA
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        
        lora_config = LoraConfig(
            r=config.lora_dim,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], # Bắt đầu với q_proj, v_proj
            lora_dropout=0.1,
            bias="none",
        )
        
        self.peft_model = get_peft_model(model, lora_config)
        self.peft_model.print_trainable_parameters()

        # 5. Khởi tạo các thành phần khác
        # logit_scale sẽ được lấy từ peft_model
        # self.logit_scale = self.peft_model.logit_scale
        
        self.loss_fct = CrossEn(config) # Sử dụng hàm loss từ dự án gốc
        
    def get_text_feat(self, input_ids, attention_mask):
        """
        Trích xuất đặc trưng văn bản.
        Ánh xạ lại API từ VTRModel gốc sang API của peft_model.
        """
        text_outputs = self.peft_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, # Cần last_hidden_state
        )
        
        # Lấy hidden state của token [CLS] từ layer cuối cùng
        # (Trong CLIP của transformers, họ dùng pooled_output, là output của CLS token qua một lớp Linear + Tanh)
        pooled_output = text_outputs[1]
        text_features = self.peft_model.text_projection(pooled_output)
        
        return text_features

    def get_video_feat(self, video, video_mask):
        """
        Trích xuất đặc trưng video.
        Giai đoạn 2: Tạm thời chỉ gọi hàm gốc, chưa tích hợp gộp token.
        """
        # video đầu vào có shape (B, F, C, H, W) từ dataloader
        # cần reshape lại thành (B*F, C, H, W)
        batch_size, num_frames, channels, height, width = video.shape
        video = video.view(-1, channels, height, width)
        
        # Trích xuất đặc trưng cho từng khung hình
        # Đây là cách tiếp cận "vanilla", không có TempMe
        image_features = self.peft_model.get_image_features(pixel_values=video)
        
        # Reshape lại và lấy trung bình
        image_features = image_features.view(batch_size, num_frames, -1)
        
        # Mean pooling
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        image_features = image_features * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(image_features, dim=1) / video_mask_un_sum

        return video_feat

    def forward(self, text_ids, text_mask, video, video_mask, idx=None, global_step=0):
        # Reshape lại cho đúng batch size
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        
        text_feat = self.get_text_feat(text_ids, text_mask)
        video_feat = self.get_video_feat(video, video_mask)
        
        # Thu thập đặc trưng từ tất cả các GPU
        if self.config.world_size > 1:
            text_feat = allgather(text_feat, self.config)
            video_feat = allgather(video_feat, self.config)
            torch.distributed.barrier()
        
        # Tính toán loss
        logit_scale = self.peft_model.logit_scale.exp()
        
        # Chuẩn hóa
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        
        t2v_logits = torch.matmul(text_feat, video_feat.t()) * logit_scale
        
        loss_t2v = self.loss_fct(t2v_logits)
        loss_v2t = self.loss_fct(t2v_logits.T)
        loss = (loss_t2v + loss_v2t) / 2
        
        return loss

    # Thêm các hàm eval để tương thích với luồng đánh giá
    def stage1_eval(self, text_ids, text_mask, video, video_mask):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])

        text_feat = self.get_text_feat(text_ids, text_mask)
        # Tạm thời chưa gộp video, nên video_feat ở đây sẽ khác
        # Ta sẽ tính mean pooling trong stage2
        
        # Để đơn giản, ta trích xuất từng frame feature
        batch_size, num_frames, channels, height, width = video.shape
        video = video.view(-1, channels, height, width)
        image_features = self.peft_model.get_image_features(pixel_values=video)
        image_features = image_features.view(batch_size, num_frames, -1)
        
        return text_feat, image_features

    def stage2_eval(self, text_feat, text_mask, video_frames_feat, video_mask):
        # Thực hiện mean pooling cho video tại đây
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_frames_feat = video_frames_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_frames_feat, dim=1) / video_mask_un_sum

        # Tính toán similarity
        logit_scale = self.peft_model.logit_scale.exp()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        t2v_logits = torch.matmul(text_feat, video_feat.t()) * logit_scale
        return t2v_logits