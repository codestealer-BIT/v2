#!/bin/bash

SESSION_NAME="my_session"

# 检查名为 my_session 的会话是否存在，不存在则创建
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME
fi

# 在指定会话中创建一个名为 "Run training" 的新窗口，并运行 top
tmux new-window -t $SESSION_NAME -n "Run training" 'cd /mnt/bn/jiny-ttls-i18n-fr1q/MMPretrain && python test.py >> output.txt'