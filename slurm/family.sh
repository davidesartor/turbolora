# model name -> family dir under outputs/runs (mirrors `family` in turbolora.models)
family() {
  case "$1" in
    deepseek*) echo deepseek ;; llama*) echo llama ;; mistral*|ministral*|mixtral*) echo mistral ;;
    qwen2.5-*-math) echo qwen2.5-math ;; qwen2.5-*-instruct) echo qwen2.5-instruct ;; qwen2.5-*) echo qwen2.5 ;;
    *) echo "unknown model family: $1" >&2; return 1 ;;
  esac
}
