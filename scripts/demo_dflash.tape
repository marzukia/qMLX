# DFlash launch demo — baseline vs DFlash, same model, same prompt
# Run: vhs scripts/demo_dflash.tape
# Requires:
#   - qMLX baseline server on :8000
#       qmlx serve qwen3.5-27b-8bit --port 8000
#   - qMLX DFlash server on :8001
#       pip install 'qmlx-serve[dflash]'
#       qmlx serve qwen3.5-27b-8bit --enable-dflash --port 8001
#   - ~60 GB free RAM (two 27B-8bit instances)

Output docs/assets/dflash-demo.gif
Output docs/assets/dflash-demo.mp4

Set Shell "zsh"
Set FontSize 15
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "python3.12 scripts/demo_dflash.py"
Enter
Sleep 90s
