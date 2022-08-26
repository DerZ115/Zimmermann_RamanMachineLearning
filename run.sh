#!/bin/bash


echo "Please make sure that the configuration file 'raman_ml.conf' is set up correctly."
read -rp "Press Enter to continue..."

# For bash linter in vscode:
# shellcheck source=./raman_ml.conf 
source ./raman_ml.conf
# shellcheck source=/dev/null
source $CONDA_DIR/etc/profile.d/conda.sh

while :
do
    echo "Choose an option:"
    echo "  0: Set up conda environment"
    echo "  1: Create dataset from individual spectra"
    echo "  2: Perform quality control"
    echo "  3: Preprocess data"
    echo "  4: Run LDA with dimensionality reduction"
    echo "  5: Run regularized linear models"
    echo "  6: Run decision-tree-based models"
    echo "all: Run steps 1-6"
    echo "or enter 'exit' to quit"

    read -r -p "Your choice: "

    if [ "${REPLY}" == "exit" ]
    then
        exit 0

    elif [ "${REPLY}" == 0 ]
    then
        if conda env list | awk '{print $1}' | grep "^$ENV_NAME$"
        then
            echo "There already exists an environment with the name $ENV_NAME."
            echo "Please choose a different name or delete the existing environment if you no longer need it."
            exit 1
        
        else
            conda create -n $ENV_NAME -f environment.yml

        fi

    else
        # Activate conda environment
        conda activate $ENV_NAME

        DATASET_OUT="$DATASET_DIR/$BASE_FILENAME.csv"
        QC_DIR="$RESULT_DIR/quality_control/"
        QC_OUT="$QC_DIR/${BASE_FILENAME}_qc.csv"
        PREP_DIR="$RESULT_DIR/preprocessing/"
        PREP_OUT="$PREP_DIR/${BASE_FILENAME}_preprocessed.csv"
        LDA_DIR="$RESULT_DIR/$BASE_FILENAME/lda_dim_reduction/"
        REG_DIR="$RESULT_DIR/$BASE_FILENAME/regularized_models/"
        TREE_DIR="$RESULT_DIR/$BASE_FILENAME/tree_based_models/"

        if [ "${REPLY}" == 1 ]
        then
            # Create dataset from individual spectra
            python ./src/01_create_dataset.py \
            -d $DIR1 $DIR2 \
            -l $LAB1 $LAB2 \
            -o $DATASET_OUT

        elif [ "${REPLY}" == 2 ]
        then
            python ./src/02_quality_control.py \
                -f $DATASET_OUT -o $QC_DIR \
                -l $QC_LIM_LOW $QC_LIM_HIGH \
                -w $QC_WINDOW -t $QC_THRESHOLD \
                -m $QC_MIN_HEIGHT -s $QC_SCORE \
                -p $QC_PEAKS -n $QC_NUM
            
        elif [ "${REPLY}" == 3 ]
        then
            python ./src/03_preprocess_data.py \
                -f $QC_OUT \
                -o $PREP_DIR \
                -l $PREP_LIM_LOW $PREP_LIM_HIGH\
                -w $PREP_WINDOW

        elif [ "${REPLY}" == 4 ]
        then
            python ./src/04_lda_dim_reduction.py \
                -f $PREP_OUT -o $LDA_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                -p "${PCA_COMP[@]}" \
                -n "${NMF_COMP[@]}" \
                -c "${FA_CLUST[@]}" \
                -d "${PEAK_DIST[@]}"

        elif [ "${REPLY}" == 5 ]
        then
            python ./src/05_regularized_models.py \
                -f $PREP_OUT -o $REG_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                --logreg-l1-c "${LR1_C[@]}" \
                --logreg-l2-c "${LR2_C[@]}" \
                --svm-l1-c "${SVM1_C[@]}" \
                --svm-l2-c "${SVM2_C[@]}"

        elif [ "${REPLY}" == 6 ]
        then
            python ./src/06_tree_based_models.py \
                -f $PREP_OUT -o $TREE_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                --tree-depth "${DT_DEPTH[@]}" \
                --tree-alpha "${DT_ALPHA[@]}" \
                --rf-depth "${RF_DEPTH[@]}" \
                --gbdt-learning-rate "${GBDT_LEARNING_RATE[@]}"
                
        elif [ "${REPLY}" == "all" ]
        then
            # Create dataset
            python ./src/01_create_dataset.py \
            -d $DIR1 $DIR2 \
            -l $LAB1 $LAB2 \
            -o $DATASET_OUT

            # Quality control
            python ./src/02_quality_control.py \
                -f $DATASET_OUT -o $QC_DIR \
                -l $QC_LIM_LOW $QC_LIM_HIGH \
                -w $QC_WINDOW -t $QC_THRESHOLD \
                -m $QC_MIN_HEIGHT -s $QC_SCORE \
                -p $QC_PEAKS -n $QC_NUM

            # Preprocessing
            python ./src/03_preprocess_data.py \
                -f $QC_OUT \
                -o $PREP_DIR \
                -l $PREP_LIM_LOW $PREP_LIM_HIGH\
                -w $PREP_WINDOW

            # LDA with dim reduction
            python ./src/04_lda_dim_reduction.py \
                -f $PREP_OUT -o $LDA_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                -p "${PCA_COMP[@]}" \
                -n "${NMF_COMP[@]}" \
                -c "${FA_CLUST[@]}" \
                -d "${PEAK_DIST[@]}"

            # Regularized models
            python ./src/05_regularized_models.py \
                -f $PREP_OUT -o $REG_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                --logreg-l1-c "${LR1_C[@]}" \
                --logreg-l2-c "${LR2_C[@]}" \
                --svm-l1-c "${SVM1_C[@]}" \
                --svm-l2-c "${SVM2_C[@]}"

            # Tree-based models
            python ./src/06_tree_based_models.py \
                -f $PREP_OUT -o $TREE_DIR \
                -s "${SCORING[@]}" -t $N_TRIALS \
                -k $N_FOLDS -j $N_CORES \
                --tree-depth "${DT_DEPTH[@]}" \
                --tree-alpha "${DT_ALPHA[@]}" \
                --rf-depth "${RF_DEPTH[@]}" \
                --gbdt-learning-rate "${GBDT_LEARNING_RATE[@]}"

        else
            echo "Please choose one of the options from the list."
        fi

    fi
done
