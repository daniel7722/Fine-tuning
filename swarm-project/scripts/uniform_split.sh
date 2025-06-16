mkdir -p data/uniform_split
for i in {0..9}; do
  mkdir -p data/uniform_split/agent_$i/train
done

TRAIN_DIR=data/imagenette/train

# Now distribute class-folders or images across agents
CLASSES=(n01440764 n02102040 n02979186 n03000684 n03028079 n03394916 n03417042 n03425413 n03445777 n03888257)
for cls in "${!CLASSES[@]}"; do
  img_paths=${TRAIN_DIR}/"${CLASSES[$cls]}"
  files=("${img_paths}"/*.JPEG)

  files=( $(printf "%s\n" "${files[@]}" | shuf) )

  # Distribute files to agents
  for idx in "${!files[@]}"; do
    agent=$(( idx % 10 ))
    cp "${files[$idx]}" "data/uniform_split/agent_${agent}/train/"
  done
done