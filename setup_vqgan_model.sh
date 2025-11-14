mkdir external && cd external
git clone https://github.com/CompVis/taming-transformers.git && git -C taming-transformers/ checkout 3ba01b2
# patch for work in Pytorch 2.x
if [[ "$OSTYPE" == "darwin"* ]]; then
    # MacOS sed command
    sed -i '' 's/from torch._six import string_classes/string_classes = str/' ./taming-transformers/taming/data/utils.py
else
    # Linux sed command
    sed -i 's/from torch._six import string_classes/string_classes = str/' ./taming-transformers/taming/data/utils.py
fi
cd ../