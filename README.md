2025.10.26   하고자 하는 과제
NewUser name newuser
vc code로 작성하여 git directory에 파일명 newuser_git_saved.txt을 업로드하고 싶음

./home이하에서 불가
/mnt/c
/mnt/d
에서 
vc code 파일 저장이 가능.
user마다 git directory 만들고 각자 push할 수 있음.

sudo nano /etc/wsl.conf
[automount]
enabled = true
options = "metadata"    
저장후 나와
