# course_script

选课脚本

# 准备

请使用python2.

1. auth

你需要准备一个auth文件，第一行写选课系统的用户名，第二行写密码

2. fateadm_auth

你需要准备一个fateadm_auth文件，第一行写用户名，第二行写密码，用于识别验证码

1. courses

你需要准备一个courses文件，第一行是学年（比如：`2016-2017-2`）；
第二行是选课开始时间（比如：2017-2-22 13:00）
接下来若干行每行是你想选的一门课，格式是`[bx/xx/rx/xwk]-课程号-课序号(X表示所有)`；
比如：xx-30240353-X
xx表示限选，其他类似
xwk表示学位课

# 运行

```sh
> python -u main.py
```