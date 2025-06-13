# create your splits directory
# mkdir -p data/splits
# for i in {0..9}; do
#   mkdir -p data/splits/agent_$i/train
#   mkdir -p data/splits/agent_$i/val
# done

# Now distribute class-folders or images across agents
CLASSES=(n01440764 n02102040 n02979186 n03000684 n03028079 n03394916 n03417042 n03425413 n03445777 n03888257)
for idx in "${!CLASSES[@]}"; do
  cls=${CLASSES[$idx]}
  # Round-robin assignment: agent = idx mod 10
  agent=$(( idx % 10 ))
  # copy train images into that agent’s train folder
  # cp data/imagenette/train/"$cls"/*.JPEG data/splits/agent_"$agent"/train/
  # likewise for val (if you want per-agent validation)
  cp data/imagenette/val/"$cls"/*.JPEG data/splits/agent_"$agent"/val/
done