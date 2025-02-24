#!/bin/bash

# 프로젝트 폴더로 이동 (사용자의 경로로 수정)
cd ~/git/expcpp || exit

# 변경 사항 추가 및 커밋
git add .
git commit -m "자동 커밋"

# 최신 코드 가져오기 (rebase 적용)
git pull origin main --rebase

# 원격 저장소로 Push
git push origin main

echo "✅ Git Push 완료!"
