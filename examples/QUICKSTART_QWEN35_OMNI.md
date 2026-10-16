# Qwen3.5-Omni 快速启动

## 环境

```bash
cd /myapp/sglang-omni
MODEL=/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315
```

> **GPU0 不可用**（跑着 Qwen3-Omni-30B 基线服务 :8008）

---

## 启动服务

### 方案 A：Thinker-Only（text/image/audio → text，1 GPU）

```bash
TORCHDYNAMO_DISABLE=1 .venv/bin/python examples/run_qwen3_5_omni_server.py \
  --model-path $MODEL \
  --gpu-thinker 1 --gpu-image-encoder 1 --gpu-audio-encoder 1 \
  --thinker-max-seq-len 32768 \
  --port 8101 --model-name qwen3_5-omni
```

### 方案 B：Speech 全量（text/image/audio → text + WAV 语音，3 GPU）

```bash
TORCHDYNAMO_DISABLE=1 .venv/bin/python examples/run_qwen3_5_omni_speech_server.py \
  --model-path $MODEL \
  --gpu-thinker 1 --gpu-talker 2 --gpu-code2wav 2 \
  --gpu-image-encoder 3 --gpu-audio-encoder 3 \
  --thinker-max-seq-len 8192 \
  --port 8101 --model-name qwen3_5-omni
```

### 后台运行

```bash
nohup bash -c 'TORCHDYNAMO_DISABLE=1 .venv/bin/python examples/run_qwen3_5_omni_speech_server.py \
  --model-path $MODEL \
  --gpu-thinker 1 --gpu-talker 2 --gpu-code2wav 2 \
  --gpu-image-encoder 3 --gpu-audio-encoder 3 \
  --thinker-max-seq-len 8192 \
  --port 8101 --model-name qwen3_5-omni 2>&1' > /tmp/speech_server.log 2>&1 &

# 查看日志
tail -f /tmp/speech_server.log

# 确认就绪
grep "Uvicorn running" /tmp/speech_server.log

# 停止
pkill -f "run_qwen3_5_omni"
```

---

## 请求示例

### 1. 纯文本对话

```bash
curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": "2+3等于几？"}],
    "max_tokens": 64
  }' | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

### 2. 图片理解

```bash
# 生成测试图片（红色方块）
python3 -c "
import base64, struct, zlib
w, h = 64, 64
raw = b''
for y in range(h):
    raw += b'\x00' + b'\xff\x00\x00' * w
def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(raw))
png += chunk(b'IEND', b'')
print(base64.b64encode(png).decode())
" > /tmp/img_b64.txt

curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$(cat /tmp/img_b64.txt)'"}},
      {"type": "text", "text": "这张图片是什么？"}
    ]}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

### 3. 音频理解（audio → text）

```bash
# 生成测试音频（440Hz 正弦波 1秒）
python3 -c "
import numpy as np, struct, base64
sr=16000; t=np.linspace(0,1,sr); samples=(np.sin(2*np.pi*440*t)*32767).astype(np.int16)
header = b'RIFF' + struct.pack('<I',36+len(samples)*2) + b'WAVEfmt '
header += struct.pack('<IHHIIHH',16,1,1,sr,sr*2,2,16) + b'data' + struct.pack('<I',len(samples)*2)
print(base64.b64encode(header + samples.tobytes()).decode())
" > /tmp/audio_b64.txt

curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": [
      {"type": "input_audio", "input_audio": {"data": "'$(cat /tmp/audio_b64.txt)'", "format": "wav"}},
      {"type": "text", "text": "描述这段音频"}
    ]}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

### 4. 语音输出（需要 Speech 服务）

```bash
# 非流式 — 一次性返回文本 + WAV 音频
curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": "用中文说你好"}],
    "max_tokens": 32,
    "modalities": ["text", "audio"]
  }' | python3 -c "
import sys, json, base64
resp = json.loads(sys.stdin.read())
msg = resp['choices'][0]['message']
print('Text:', msg.get('content', ''))
audio = msg.get('audio', {})
if audio:
    raw = base64.b64decode(audio['data'])
    print(f'Audio: {len(raw)} bytes WAV')
    with open('/tmp/output.wav', 'wb') as f: f.write(raw)
    print('Saved: /tmp/output.wav')
else:
    print('No audio (需要 Speech 服务)')
"
```

```bash
# 流式 — 文本逐 token 流出，最后带 audio chunk
curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": "Count 1 to 5"}],
    "max_tokens": 32,
    "stream": true,
    "modalities": ["text", "audio"]
  }'
```

### 5. 只要文本不要语音（即使是 Speech 服务）

```bash
curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "modalities": ["text"]
  }' | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

### 6. 图片 + 音频 组合输入

```bash
curl -s http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-omni",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$(cat /tmp/img_b64.txt)'"}},
      {"type": "input_audio", "input_audio": {"data": "'$(cat /tmp/audio_b64.txt)'", "format": "wav"}},
      {"type": "text", "text": "同时描述图片和音频"}
    ]}],
    "max_tokens": 200
  }' | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

---

## GPU 分配一览

| 方案 | GPU 数 | 分配 | thinker seq_len |
|------|--------|------|-----------------|
| Thinker-only | 1 | 全部同卡 | 32768 |
| Speech | 3 | thinker=1, talker+c2w=2, enc=3 | 8192 |

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `Not enough memory` | 23B 模型 ~46GB，KV cache 空间不足 | 降 `--thinker-max-seq-len` 或 encoder 分卡 |
| Stream 只有文本没有 finish | talker 内部报错 | 看 log: `grep ERROR /tmp/speech_server.log` |
| 启动卡住不动 | 上次进程未清理占着 GPU | `pkill -9 -f multiprocessing.spawn && sleep 2` |
| `CUDA error: device-side assert` | 前一次进程异常退出污染 GPU 状态 | 等几秒让 GPU reset，或换一张卡 |
