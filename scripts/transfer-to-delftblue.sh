echo "$PWD"

mkdir temp
cp -r src temp
cp -r auxiliary temp
cp requirements.txt temp
cp scripts temp

read -p "Enter netid pls: " name
scp -pr temp "$name"@login.delftblue.tudelft.nl:/home/"$name"/Project

rm -r temp