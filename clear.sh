#!/usr/bin/env sh
if [ -e "./logs/" ];then
rm -r -f logs
echo "logs已经删除"
else
echo "logs不存在"
fi

if [ -e "./checkpoint/" ];then
rm -r -f checkpoint
echo "checkpoint已经删除"
else
echo "checkpoint不存在"
fi