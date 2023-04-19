# check if Python virtual environment env exists
if [ ! -f "env/bin/activate" ]; then
    echo "Environment env does not exist. Please install it via ./install.sh"
    exit 1
fi

# activate Python virtual environment
source env/bin/activate

# create output folder
rm -rf models/
mkdir models

# train a model for part 1 (regression-based ANN) and evaluate it
echo "#########   PART 1: regression-based model ################"
python scripts/train.py --model-type regresion_nn --data data/train/ --save models/part1.pt 
python scripts/evaluate.py --model models/part1.pt --data data/test/
echo ""

# train a model for part 2 - basic (classification-based ANN) and evaluate it
echo "#########   PART 2 (basic): classification-based model ################"
python scripts/train.py --model-type classification_nn --data data/train/ --save models/part2_basic.pt
python scripts/evaluate.py --model models/part2_basic.pt --data data/test/
echo ""

echo "#########   PART 2 (advanced): cost-sensitive classification-based model ################"
python scripts/train.py --model-type classification_nn_cost --data data/train/ --save models/part2_advanced.pt 
python scripts/evaluate.py --model models/part2_advanced.pt --data data/test/
echo ""

# train a model for part 3 - extension 1 and evaluate it
echo "#########   PART 3 (extension 1): pairwise cost-sensitive classification model ################"
# YOUR CODE HERE: please add commands for each extension following the same template as above. 
# REMEMBER TO PRINT OUT THE DESCRIPTION (those "echo ..." lines) before every pair of train and evaluate command).


# deactivate Python virtual environment
deactivate
