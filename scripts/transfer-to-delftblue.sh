echo "$PWD"

mkdir temp
cp -r src temp
cp -r auxiliary temp
cp requirements.txt temp
cp -r scripts temp
cp scripts/run_job.sh temp

read -p "Enter netid pls: " name
scp -pr temp "$name"@login.delftblue.tudelft.nl:/home/"$name"/Project

rm -r temp