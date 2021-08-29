#!bin/bash
path_to_folder_temp=$KC4ALIGN'/temp'$(( ( RANDOM % 100 )  + 1 ))
while [ -d "$path_to_folder_temp" ]
do
    path_to_folder_temp=$KC4ALIGN'/temp'$(( ( RANDOM % 100 )  + 1 ))
done
mkdir $path_to_folder_temp
echo $path_to_folder_temp
cd $KC4ALIGN
./overlap.py -i $1 -o $path_to_folder_temp/overlaps.vi -n 4 -l vi
./overlap.py -i $2 -o $path_to_folder_temp/overlaps.lo.trans2vi -n 4 -l vi

cd $LASER'/tasks/embed'
./embed.sh $path_to_folder_temp/overlaps.vi vi $path_to_folder_temp/overlaps.vi.emb
./embed.sh $path_to_folder_temp/overlaps.lo.trans2vi vi $path_to_folder_temp/overlaps.lo.trans2vi.emb

cd $KC4ALIGN
./vecalign.py --alignment_max_size 2 --src $1 --tgt $2 --src_embed $path_to_folder_temp/overlaps.vi $path_to_folder_temp/overlaps.vi.emb --tgt_embed $path_to_folder_temp/overlaps.lo.trans2vi $path_to_folder_temp/overlaps.lo.trans2vi.emb --mode align_sentence --vis_raw_file $4 --los_raw_file $5 > $3

rm -r "$path_to_folder_temp"
