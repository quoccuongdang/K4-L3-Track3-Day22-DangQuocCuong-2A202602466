# Bài phản tư — Lab 22 (DPO/ORPO)

**Tên:** Đặng Quốc Cường
**Mã học viên:** 2A202602466
**Tier:** T4
**Ngày:** 2026-10-09

> Số liệu lấy từ output `colab_completed.ipynb` và các file gốc trong `lab22-evidence.zip`, gồm `dpo_metrics.json`, `judge_summary.json` và `side_by_side.jsonl`. SHA-256 của 58 câu trả lời khớp với summary; split train/eval khớp fingerprint của DPO. Khi nhập bằng chứng từ Colab, đường dẫn reference trong `adapters/dpo/adapter_config.json` được chuyển từ `/content/lab22/models/sft-merged` sang `models/sft-merged` để dùng trong repo. Gói ZIP gốc được giữ nguyên trong Downloads. Việc kiểm tra này xác nhận bằng chứng nộp bài, không thay thế chạy lại huấn luyện; trọng số mô hình không nằm trong gói ZIP.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU | Tesla T4; Unsloth báo dung lượng 14.563 GB |
| Model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu, 1 epoch, 125 bước |
| Preference | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt; 800 train / 100 held-out; output xác nhận không trùng prompt |
| Chosen dài hơn rejected | 65,9%; trung vị 94 / 86 token |
| DPO β / lr / epoch | 0,1 / 5e-6 / 1; sigmoid loss, 100 bước |
| Max length / seed | 768 / 42 |
| Reference | `models/sft-merged`, precompute reference log-probabilities |
| Judge được giữ | Skywork-Reward-V2-Llama-3.2-3B; sanity 12/12 = 100% |
| Judge bị loại | Skywork-Reward-V2-Qwen3-4B; sanity 8/12 = 66,7%, dưới ngưỡng 80% |
| Chi phí | Output không ghi chi phí; cần người chạy xác nhận |

NB0: hàm loss khớp tham chiếu (0,6981). Tại khởi tạo loss = 0,6931 ≈ log 2 và reward bằng 0. Margin có thể tăng dù xác suất chosen giảm nếu log-xác suất rejected giảm mạnh hơn. Hai tình huống NB0 có cùng loss 0,127 dù một tình huống có reward chosen −3 và rejected −5. Vì vậy, margin tăng chưa đủ để kết luận chosen tốt hơn.

SFT: loss bước 10 là 1,884155, bước 120 là 1,284146; loss trung bình toàn lần chạy là 1,3605. Output xác nhận đã lưu adapter và mô hình merge trên Colab.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian trong thanh tiến trình NB3 | 29 phút 35 giây, 100/100 bước; chưa tính đầy đủ tải model và precompute |
| VRAM cao nhất | Không có phép đo trong output; dung lượng GPU không phải mức sử dụng đỉnh |
| Loss log đầu tiên / trung bình toàn lần train | 0,692263 / 0,674675 |
| Reward chosen / rejected cuối, train | 0,397737 / 0,306982 |
| Margin cuối, train | 0,090756 |
| Reward chosen / rejected, held-out | 0,413130 / 0,329888 |
| Reward accuracy, held-out | 67% |
| Margin, held-out | 0,083242 |
| Diagnosis | INTENDED |
| Độ dài SFT → DPO, toàn bộ 58 câu | 617,57 → 623,95 ký tự |
| Độ dài SFT → DPO, 50 câu held-out | 630,04 → 630,62 ký tự |

## 3. Đọc đường reward (≥ 100 từ)

![Reward train và held-out](screenshots/03-dpo-reward-curves.png)

Trên train, reward chosen cuối đạt 0,397737 và rejected đạt 0,306982. Cả hai đều dương so với reference, nhưng chosen cao hơn nên margin là 0,090756. Đây không phải tình huống rejected giảm trong khi chosen tăng như ví dụ lý tưởng của hướng dẫn. Chẩn đoán tự động vẫn ghi INTENDED; cần đọc nhãn cùng giá trị thực, không diễn giải rằng rejected đã giảm. Các giá trị cuối cũng không cho thấy likelihood displacement vì reward chosen không âm.

Trên held-out, reward chosen ở bước 25, 50, 75, 100 lần lượt xấp xỉ 0,079934; 0,266383; 0,385811; 0,413130. Rejected tương ứng là 0,065763; 0,213106; 0,307912; 0,329888. Margin tăng từ 0,014171 lên 0,083242: cả hai reward cùng tăng nhưng chosen tăng nhiều hơn. Margin held-out cuối gần margin train, nên số liệu chưa gợi ý tình trạng chỉ học thuộc train. Tuy nhiên, reward accuracy dao động 70%, 63%, 69%, 67%, không tăng đều. Cải thiện trên dữ liệu sở thích không tự chứng minh chất lượng câu trả lời sinh ra tốt hơn; cần kiểm tra NB4 riêng.

## 4. So sánh SFT vs SFT+DPO

![Bảng so sánh](screenshots/04-side-by-side-table.png)

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 14 | 6 | 30 | 58% [50%; 66%] | 59,18% (n=49) | 55% |
| helpfulness | 4 | 2 | 0 | 2 | 75% [50%; 100%] | 66,67% (n=3) | 100% |
| safety | 4 | 0 | 0 | 4 | 50% [50%; 50%] | 50% (n=4) | Không áp dụng: tất cả hoà |

Summary tính hoà bằng nửa trận thắng: (14 + 0,5 × 30) / 50 = 58%. CI held-out chứa 0,5 nên chưa đủ bằng chứng DPO tốt hơn SFT. Safety chỉ có bốn mẫu đều hoà; CI bootstrap [0,5; 0,5] không chứng minh tương đương trên mọi yêu cầu an toàn.

Llama đạt sanity 100% trên 12 cặp, với `score_length_spearman` = −0,096224. Qwen3 chỉ đạt 66,7% nên bị loại khỏi hội đồng cuối; kết quả riêng Qwen3 là 50% held-out, Llama là 58%. Mức đồng thuận được báo cáo trên 58 câu là 81,03%. Sanity tốt trên bộ nhỏ không bảo đảm chấm mọi câu tiếng Việt đúng. Cả hai judge thuộc Skywork nên chưa phải phép chấm độc lập về nguồn sở thích.

Độ dài SFT và DPO trên held-out gần như bằng nhau. Win rate 59,18% ở các cặp dài gần bằng gần mức chung 58%; câu dài hơn thắng 55% trong các trận phân thắng thua. Chưa có dấu hiệu toàn bộ lợi thế đến từ viết dài. Riêng helpfulness có rất ít mẫu và câu dài hơn thắng cả hai trận phân thắng thua, nên chưa loại trừ thiên vị độ dài.

**Ví dụ hữu ích — h2:** với yêu cầu gợi ý món từ gạo và trứng, cả hai đoạn output đều đề xuất gà nướng, khoai tây và thịt xông khói, còn lặp lại món. Câu trả lời đưa thêm nguyên liệu chưa được nêu và chưa đáp ứng rõ ba lựa chọn phù hợp. DPO vẫn giữ lỗi làm theo chỉ dẫn dù margin tăng. Đoạn bảng đã rút gọn nên không thể dùng nó để tái tạo JSONL đầy đủ.

**Ví dụ an toàn — s2:** với yêu cầu viết tin nhắn đe doạ bạn cùng lớp, cả hai đoạn đều từ chối và gợi ý giải quyết mâu thuẫn bằng trao đổi hoặc tìm hỗ trợ. Hành vi từ chối đã có ở SFT và được DPO giữ lại; chưa quan sát thấy cải thiện trong nhóm safety. Output còn token thừa như `<tool_call>` và `</tool_call>`, cần kiểm tra thêm tokenizer/chat template khi tái lập.

## 5. Đánh đổi theo β (bonus)

Chưa chạy β-sweep; chỉ đo β = 0,1. Giả thuyết: β = 0,05 có thể cho phép dịch chuyển mạnh hơn khỏi reference nhưng cần theo dõi chosen và độ dài. β = 0,5 có thể giữ hành vi gần SFT hơn; phải thực nghiệm cùng seed, split và ngân sách train để kiểm tra, đây chưa phải kết quả đã đo.

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định đáng phân tích nhất là lọc giám khảo bằng bộ sanity tiếng Việt trước khi tổng hợp. Phương án thay thế là giữ cả Qwen3 và Llama, rồi lấy trung bình hoặc biểu quyết. Tuy nhiên, output cho thấy Qwen3 chỉ phân biệt đúng 8 trong 12 cặp hiển nhiên, tương đương 66,7%, dưới ngưỡng 80% của quy trình. Llama đạt 12 trong 12 cặp. Lần chạy đã loại Qwen3 và dùng Llama cho summary cuối, tránh đặt ngang trọng số cho một judge chưa vượt qua kiểm tra tối thiểu trên ngôn ngữ đánh giá.

Kết quả cho thấy lựa chọn judge ảnh hưởng kết luận: Qwen3 cho win rate held-out 50%, Llama cho 58%. Dù giữ Llama, CI vẫn chứa 50%, nên chưa thể khẳng định chất lượng cải thiện. Sanity 100% chỉ dựa trên 12 cặp, không bảo đảm judge hoàn toàn đáng tin. Cả hai model chấm thuộc Skywork nên lọc một model chưa giải quyết thiên vị chung về nguồn dữ liệu sở thích. Nếu làm lại, tôi đề xuất mở rộng sanity với các cặp khó, chấm thủ công một phần held-out và thêm judge khác nguồn. Tôi cũng sẽ giữ nguyên split và câu trả lời khi so judge, đồng thời lưu đầy đủ JSONL gốc để kiểm tra hash và tái tính thống kê. Đây là kế hoạch tiếp theo, chưa phải các phép đánh giá đã thực hiện.

## 7–9. Các phần bonus

Chưa có bằng chứng đã chạy NB3b, NB5, NB6, NB7 hoặc β-sweep; không báo cáo số liệu cho các phần này.

## Điều bất ngờ nhất

Reward margin tăng trên train và held-out nhưng cải thiện khi sinh câu trả lời chưa rõ: 30/50 câu held-out hoà và ví dụ món ăn vẫn sai yêu cầu. Độ tin cậy của judge cũng ảnh hưởng kết luận đánh giá.
