#### Python example instructions

clone the [github](https://github.com/balajiln/mondrianforest) then

1. run mondrianforest/process_data/commands.sh to download the data sets and process them for python.
2. Then copy the folder process_data and one dir above the full mondrianforest dir (their package requires this).
3. go to mondrianforest/src and copy python-example.py to this dir
4. make everything executable in bash
5. run the command ./python-example.py --dataset letter --n_mondrians 1 --budget -1 --normalize_features 1 --optype class --draw_mondrian 0 --n_minibatches 1





All of the on-line algorithms are in *Mondrian_Forest.jl*

There are copies of the *Mondrian_Tree.jl*, *Mondrian_Forest_Classifier.jl* , and *Axis_Aligned_Box.jl* in this
directory so these can be changed however.
