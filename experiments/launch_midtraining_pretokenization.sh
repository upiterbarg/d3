folder_containg_source_data=./my_source
destination_directory=./my_dest
tokenizer_name_or_path=???
tokenizer_eos_id=128001
tokenizer_pad_token_id=128001
num_cpu_cores=8
dtype=uint32 ## dtype must be set to accomodate your tokenizer vocabulary in order to avoid errors

generation_command=(
	dolma tokens \
	--documents ${folder_containg_source_data} \
	--destination ${destination_directory} \
	--tokenizer.name_or_path ${tokenizer_name_or_path} \
	--tokenizer.eos_token_id ${tokenizer_eos_id} \
	--tokenizer.pad_token_id ${tokenizer_pad_token_id} \
	--processes ${num_cpu_cores} \
	--dtype ${dtype} \
	$@
)
log_file=./where_to_log_stdout
"${generation_command[@]}" | tee -a "${log_file}.txt"
