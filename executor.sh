
# BEGIN: Loop and move files
count=0
for file in ./dataset128/*; do
    if [ $count -lt 10 ]; then
        mv "$file" ./datasetsm
        count=$((count+1))
    else
        break
    fi
done
# END: Loop and move files
