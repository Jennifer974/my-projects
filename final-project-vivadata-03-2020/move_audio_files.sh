#!/bin/bash
for celeb_id in `ls ../../../../Documents/Final-Project-data/vox1_dev_wav/wav`
do
	for lst_audio in `ls ../../../../Documents/Final-Project-data/vox1_dev_wav/wav/$celeb_id`
	do
		for audio in  `ls ../../../../Documents/Final-Project-data/vox1_dev_wav/wav/$celeb_id/$lst_audio`
		do
			cp ../../../../Documents/Final-Project-data/vox1_dev_wav/wav/$celeb_id/$lst_audio/$audio ../../../../Documents/Final-Project-data/vox1_dev_wav/audio_files/$celeb_id-$lst_audio-$audio		
		done


	done
done
