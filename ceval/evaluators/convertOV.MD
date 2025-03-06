optimum-cli export openvino --model ../DeepSeek-R1-Distill-Llama-8B --task text-generation-with-past --weight-format int4 --sym --group-size 128 --ratio 0.8 DeepSeek-R1-Distill-Llama-8B/INT4_compressed_weights

Mixed-Precision assignment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 224/224 • 0:01:13 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_asym                 | 30% (70 / 226)              | 20% (68 / 224)                         |
+---------------------------+-----------------------------+----------------------------------------+
| int4_sym                  | 70% (156 / 226)             | 80% (156 / 224)                        |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:20 • 0:00:00


optimum-cli export openvino --model ../DeepSeek-R1-Distill-Llama-8B --task text-generation-with-past --weight-format int4 --sym --group-size 128 --ratio 0.8 --awq --dataset wikitext2 --num-samples 128 DeepSeek-R1-Distill-Llama-8B/INT4-AWQ_compressed_weights


Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:07:15 • 0:00:00
Mixed-Precision assignment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 224/224 • 0:01:00 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_asym                 | 31% (97 / 226)              | 21% (95 / 224)                         |
+---------------------------+-----------------------------+----------------------------------------+
| int4_sym                  | 69% (129 / 226)             | 79% (129 / 224)                        |
+---------------------------+-----------------------------+----------------------------------------+
Applying AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 29/29 • 0:02:45 • 0:00:00
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:37 • 0:00:00

optimum-cli export openvino --model ../DeepSeek-R1-Distill-Llama-8B --task text-generation-with-past --weight-format int4 --group-size -1 --ratio 1.0 --sym DeepSeek-R1-Distill-Llama-8B\INT4-NPU_compressed_weights

INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_asym                 | 13% (2 / 226)               | 0% (0 / 224)                           |
+---------------------------+-----------------------------+----------------------------------------+
| int4_sym                  | 87% (224 / 226)             | 100% (224 / 224)                       |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:28 • 0:00:00

optimum-cli export openvino --model ../DeepSeek-R1-Distill-Llama-8B --task text-generation-with-past --weight-format int8 DeepSeek-R1-Distill-Llama-8B\INT8_compressed_weights

INFO:nncf:Statistics of the bitwidth distribution:
+---------------------------+-----------------------------+----------------------------------------+
| Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
+===========================+=============================+========================================+
| int8_asym                 | 100% (226 / 226)            | 100% (226 / 226)                       |
+---------------------------+-----------------------------+----------------------------------------+
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:27 • 0:00:00

optimum-cli export openvino --model ../DeepSeek-R1-Distill-Llama-8B --task text-generation-with-past --weight-format fp16 DeepSeek-R1-Distill-Llama-8B\FP16

