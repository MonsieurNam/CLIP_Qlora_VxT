import torch
import gradio as gr
import argparse
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import os
import json
import glob

# Import các lớp cần thiết từ dự án của bạn
from tvr.models.modeling import VTRModel
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.dataloader_retrieval import RetrievalDataset

# --- START: Sửa lỗi NotImplementedError ---
# Tạo một lớp Dataset đơn giản cho demo, kế thừa từ lớp cơ sở
# để có thể sử dụng lại hàm _get_rawvideo_dec.
class DemoDataset(RetrievalDataset):
    def _get_anns(self, subset='test'):
        # Chúng ta không cần annotation cho demo,
        # vì vậy hàm này chỉ cần trả về các dict rỗng.
        return {}, {}
# --- END: Sửa lỗi NotImplementedError ---

# --- Các hàm Helper để xử lý text ---
SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                 "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

def _tokenize_text(tokenizer, text, max_words):
    words = tokenizer.tokenize(text)
    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), torch.tensor(input_mask).unsqueeze(0)

def get_args():
    parser = argparse.ArgumentParser(description="Text-Video Retrieval Demo")
    
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default="modules/")
    parser.add_argument("--output_dir", default="demo_output", type=str)

    parser.add_argument('--base_encoder', type=str, default="ViT-B/32")
    parser.add_argument('--max_words', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--video_framerate', type=int, default=1)
    parser.add_argument('--lora_dim', type=int, default=8)
    parser.add_argument('--tome_r', type=int, default=2)
    parser.add_argument('--merge_layer', type=str, default='8-9-10')
    parser.add_argument('--merge_frame_num', type=str, default='2-2-3')
    parser.add_argument('--merge_token_proportion', type=str, default='30-10')
    parser.add_argument('--frame_pos', type=int, default=1)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--tome_tracesource", type=bool, default=False)
    parser.add_argument("--tome_propattn", type=bool, default=True)
    parser.add_argument("--init_model", default=None, type=str)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.init_model = args.checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải model...")
    tokenizer = ClipTokenizer()
    model = VTRModel(args)
    
    model_state_dict = torch.load(args.init_model, map_location='cpu')
    model.load_state_dict(model_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    print("Tải model thành công!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    feature_cache_path = join(args.output_dir, "video_features.pt")
    paths_cache_path = join(args.output_dir, "video_paths.json")

    if exists(feature_cache_path) and exists(paths_cache_path):
        print(f"Đang tải các đặc trưng video đã được cache từ {args.output_dir}...")
        all_video_feats = torch.load(feature_cache_path, map_location=device)
        with open(paths_cache_path, 'r') as f:
            all_video_paths = json.load(f)
        print("Tải cache thành công!")
    else:
        print("Không tìm thấy cache. Bắt đầu trích xuất đặc trưng từ thư mục video...")
        
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files_to_index = []
        for ext in video_extensions:
            video_files_to_index.extend(glob.glob(join(args.video_path, ext)))
        
        if not video_files_to_index:
            raise FileNotFoundError(f"Không tìm thấy file video nào trong: {args.video_path}")
            
        print(f"Tìm thấy {len(video_files_to_index)} video để lập chỉ mục.")
        
        # --- START: Sửa lỗi NotImplementedError ---
        # Sử dụng DemoDataset thay vì RetrievalDataset
        temp_dataset = DemoDataset(
            subset='test', anno_path=None, video_path=args.video_path,
            tokenizer=tokenizer, max_frames=args.max_frames,
            video_framerate=args.video_framerate, config=args
        )
        # --- END: Sửa lỗi NotImplementedError ---
        
        temp_dataset.video_dict = {os.path.basename(p): p for p in video_files_to_index}

        all_video_feats_list = []
        all_video_paths = []

        with torch.no_grad():
            for video_path in tqdm(video_files_to_index, desc="Đang trích xuất đặc trưng video"):
                video_id = os.path.basename(video_path)
                video_tensor, video_mask_tensor = temp_dataset._get_rawvideo_dec(video_id)
                video_tensor = torch.from_numpy(video_tensor).to(device)
                video_mask_tensor = torch.from_numpy(video_mask_tensor).unsqueeze(0).to(device)
                
                video_feat = model.get_video_feat(video_tensor, video_mask_tensor)
                all_video_feats_list.append(video_feat)
                all_video_paths.append(video_path)

        all_video_feats = torch.cat(all_video_feats_list, dim=0)
        
        print(f"Đang lưu cache đặc trưng vào {args.output_dir}...")
        torch.save(all_video_feats, feature_cache_path)
        with open(paths_cache_path, 'w') as f:
            json.dump(all_video_paths, f)
        print("Lưu cache thành công!")

    all_video_feats /= all_video_feats.norm(dim=-1, keepdim=True)
    print(f"Đã lập chỉ mục {len(all_video_paths)} video. Demo đã sẵn sàng!")
    
    def retrieve_videos(text_prompt, top_k=10):
        print(f"\nNhận được truy vấn: '{text_prompt}'")
        if not text_prompt:
            return None, None
        with torch.no_grad():
            text_ids, text_mask = _tokenize_text(tokenizer, text_prompt, args.max_words)
            text_ids, text_mask = text_ids.to(device), text_mask.to(device)
            text_feat = model.get_text_feat(text_ids, text_mask)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            similarities = torch.matmul(text_feat, all_video_feats.T).squeeze(0)
            top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(all_video_paths)))
            top_1_video_path = all_video_paths[top_indices[0].item()]
            top_k_video_paths = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                score = top_scores[i].item()
                path = all_video_paths[idx]
                top_k_video_paths.append((path, f"Rank {i+1}\nScore: {score:.3f}"))
            print(f"Video phù hợp nhất (Top 1): {top_1_video_path}")
            print(f"Top {len(top_k_video_paths)} kết quả đã được tìm thấy.")
            return top_1_video_path, top_k_video_paths

    with gr.Blocks() as iface:
        gr.Markdown(
            """
            # Demo Tìm kiếm Video bằng Văn bản (TempMe)
            Nhập một câu mô tả (bằng tiếng Anh) và hệ thống sẽ tìm video phù hợp nhất từ bộ sưu tập video được cung cấp.
            """
        )
        with gr.Row():
            text_prompt = gr.Textbox(lines=2, placeholder="Nhập mô tả video ở đây...", label="Mô tả Video (Prompt)")
        with gr.Row():
            submit_btn = gr.Button("Tìm kiếm", variant="primary")
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Kết quả phù hợp nhất (Top 1)")
                top_1_output = gr.Video(label="Top 1 Video")
            with gr.Column(scale=2):
                gr.Markdown("## Top 10 kết quả")
                top_10_output = gr.Gallery(label="Top 10 Videos", columns=5, object_fit="contain", height="auto")
        submit_btn.click(fn=retrieve_videos, inputs=text_prompt, outputs=[top_1_output, top_10_output])
        gr.Examples(
            examples=[
                ["a man is talking"],
                ["a girl is cooking in the kitchen"],
                ["cartoon characters are fighting"],
                ["a train is moving on the rail"],
                ["someone is playing piano"]
            ],
            inputs=text_prompt
        )
    
    iface.launch(share=True, allowed_paths=[args.video_path])

if __name__ == "__main__":
    main()