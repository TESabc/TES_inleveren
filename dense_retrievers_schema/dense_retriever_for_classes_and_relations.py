import sys

from safetensors import torch
import sklearn  # need to be installed

sys.path.append('.')
sys.path.append('..')
import optuna  # need to install
import os
from tqdm import tqdm  # this is something you need to install i think. Or just remove from the script.
import argparse
import json
import os.path
import random

import numpy as np
import torch  # this needs to be installed
from datasets import load_metric  # this needs to be installed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, IntervalStrategy
from dataloader.kkg_json_loader import KadasterJsonLoader
import certifi

# this one you need to fix maybe!
# Gets the reverse relations basically
# from retriever.freebase_retriever import FreebaseRetriever


from utils.file_util import read_list_file
from utils.hugging_face_dataset import HFDataset
import accelerate


class KadasterSchemaDenseRetriever:
    def __init__(self):
        # We assign a tokenizer needed to chunk text for transformers.
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # The number of negative samples that will be generated within the training data of the dense retrievers
        self.num_negative = 20
        # The strategy type for sampling negative samples
        self.negative_strategy = 'random_question'
        random.seed(429)

        # we read all the classes and relations in the kadaster knowledge graph
        self.all_class = read_list_file('kkg_classes.txt')
        self.all_relation = read_list_file('kkg_relations.txt')

        # We create dataloader objects for our different datasets
        self.production_training_data = KadasterJsonLoader(
            r'..\Training_data\Main_Training_data\training_data_brt_filter.json')
        self.train_dev_split_training_data = KadasterJsonLoader(
            r'..\Training_data\Train_dev_split_for_hyperparameter_tuning\train_data_brt_filter.json')
        self.train_dev_split_dev_data = KadasterJsonLoader(
            r'..\Training_data\Train_dev_split_for_hyperparameter_tuning\dev_data_brt_filter.json')

        # This will later be used for relations that go the other way around..
        # self.retriever = FreebaseRetriever()

    '''
    Note that the following code creates positive and negative samples. This is needed for a classification task also
    called:
    "Supervised Learning with Positive and Negative Examples" or "Binary Classification."
    '''

    def encode(self, dataloader, output_dir, schema_type):
        # We create paths for the encodings and labels
        encoding_path = output_dir + '_' + schema_type + '_encodings'
        labels_path = output_dir + '_' + schema_type + '_labels'

        # We check if the encodings are already created. If they are created we load them into torch and return them
        # and stop the remainder of the function.
        if output_dir is not None:
            if os.path.isfile(encoding_path) and os.path.isfile(labels_path):
                return torch.load(encoding_path), torch.load(labels_path)

        # This is simply where we will store the data
        text_a = []  # question
        text_b = []  # schema
        labels = []

        # We loop over all the data in the dataloader object
        for idx in tqdm(range(0, dataloader.len)):
            # We get the question
            question = dataloader.get_question_by_idx(idx)
            # We get the golden class
            golden_class = dataloader.get_golden_class_by_idx(idx)
            # We get the golden relation
            golden_relation = dataloader.get_golden_relation_by_idx(idx)

            # THIS PART WE DON'T HAVE YET!! NEED TO INVESTIGATE WHETHER WE CAN IMPLEMENT THIS AND WHETHER WE NEED IT
            # We get the golden reverse relations (probably simply the ones moving in the opposite direction)
            # golden_reverse_relation = self.retriever.reverse_relation_list(golden_relation)

            # 1. positive samples
            # we select the correct schema items based on which schema type we are treating
            golden_schema = golden_class if schema_type == 'class' else golden_relation
            # We simply create positive samples for each of the correct schemas belonging to the question
            # For each schema item we create a positive sample with label 1
            for s in golden_schema:
                text_a.append(question)
                text_b.append(s)
                labels.append(1)

            # 2. negative samples
            # There are two approaches for getting negative samples. Either by randomly selecting from the entire list,
            # or by randomly selecting a question.
            negative_class = []
            negative_relation = []

            # with the 'random' approach we simply select self.num_negative classes and relations from the entire list
            if self.negative_strategy == 'random':
                negative_class = random.sample(self.all_class, self.num_negative)
                negative_relation = random.sample(self.all_relation, self.num_negative)

            # simply choose a question which is not the question at hand here, and we append the negative class
            # and negative relations to our list
            elif self.negative_strategy == 'random_question':
                for i in range(self.num_negative):
                    # select a random index from all the indices in our dataset
                    random_idx = random.randrange(0, dataloader.len)
                    # we keep selecting a random index until we select one which is not the current question
                    while random_idx == idx:
                        random_idx = random.randrange(0, dataloader.len)
                    # we add all the associated schema items to our main list
                    negative_class.extend(dataloader.get_golden_class_by_idx(random_idx))
                    negative_relation.extend(dataloader.get_golden_relation_by_idx(random_idx))

            # We put the appropriate list in the 'negative_schema' variable
            negative_schema = negative_class if schema_type == 'class' else negative_relation

            # We loop through the negative schema items while ensuring that they are not in the golden schema
            # or in the reverse golden schema
            for s in negative_schema:
                if s not in golden_schema:  # and s not in golden_reverse_relation:
                    text_a.append(question)
                    text_b.append(s)
                    labels.append(0)
        # end for each question

        # We create encodings (note these are not the same as embeddings, and we return them as pytorch tensors)
        encodings = self.tokenizer(text_a, text_b, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # We save the encodings. (so if we run this function again it can retrieve them and not create them again)
        torch.save(encodings, encoding_path)
        torch.save(labels, labels_path)
        # the function returns the encodings and labels (whether they were created or loaded from disk)
        return encodings, labels

    def train_with_validation_to_determine_epochs(self, schema_type,
                                                  output_dir='../saved_models/saved_models_with_validation'):
        # we put some output_dir here where the model will be saved
        output_dir = output_dir + '/' + schema_type

        # We check whether a file already exists, if it exists this function gets skipped
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('[INFO] Model already exists, skip training')
            return

        # We obtain the encodings for the train and dev data
        train_encodings, train_labels = self.encode(self.train_dev_split_training_data,
                                                    output_dir=output_dir + '/train',
                                                    schema_type=schema_type)
        dev_encodings, dev_labels = self.encode(self.train_dev_split_dev_data, output_dir=output_dir + '/dev',
                                                schema_type=schema_type)

        # Basically a custom object storing the encodings and labels
        train_dataset = HFDataset(train_encodings, train_labels)
        dev_dataset = HFDataset(dev_encodings, dev_labels)

        # training settings
        '''
        The datasets library is part of the Hugging Face ecosystem and provides functionalities for working with 
        various datasets commonly used in natural language processing (NLP) and machine learning. 
        The load_metric function specifically serves to load evaluation metrics that can be used for assessing 
        the performance of models on specific tasks or datasets.

        In this case we will be using accuracy to see how well it does for the ground truth

        Note that self.model_name = 'bert-base-uncased'
        We simply create this model with 2 labels and set it to training mode.

        setting the model to training mode, allows it to update its weights during the training process. 
        This means the model is prepared to learn from the data during subsequent training steps.
        '''
        metric = load_metric("accuracy")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.train()

        '''
        Simply takes all the predictions and compares them with the true labels and 
        then applies accuracy performance metric.
        '''

        def compute_metrics(eval_pred):
            # we have both model predictions and ground truth labels
            '''
            logits typically represent the raw outputs or scores produced by the model for each class.
            In this case, it's assumed to contain the output logits for each class from the
            sequence classification model.
            '''
            logits, labels = eval_pred

            # Basically we take the index of the maximum amongst the classes
            # These ofcourse, represent our predictions
            predictions = np.argmax(logits, axis=-1)
            # Now we simply compute acuracy
            return metric.compute(predictions=predictions, references=labels)

        # This is class from Hugging Face that holds training settings for models
        '''
        output_dir: Specifies the directory path where model checkpoints, logs, and outputs will be saved during 
        and after training.

        do_train: means that we will use the model for training

        do_eval: means evaluation will also be performed on the evaluation set

        do_predict: This means predictions will be done on the test set

        per_device_eval_batch_size: Defines the batch size used for training per GPU/device. 
        Larger batch sizes can improve training speed but may require more memory.

        num_train_epochs: one epoch represents one entire cycle through the training data.
        The algorithm has seen and learned from every sample in the training data in one epoch. An epoch can consist
        of multiple iterations. The iterations use mini-batches of training data to update model parameters

        learning_rate: step size during optimization

        evaluation_strategy: this argument determines when the evaluation (validation) of the model is 
        performed during training. With out current setting we evaluate after every epoch.

        save_strategy: Determines strategy for saving checkpoints during training. Currently after each epoch
        we save checkpoints. Note the checkpoints are basically the model's current state.

        load_best_model_at_end: This boolean flag specifies whether to load the best model based on 
        evaluation results at the end of training.
        '''

        training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                          per_device_train_batch_size=256, per_device_eval_batch_size=64,
                                          num_train_epochs=3, learning_rate=5e-5,
                                          evaluation_strategy=IntervalStrategy.EPOCH,
                                          save_strategy=IntervalStrategy.EPOCH, load_best_model_at_end=True,
                                          logging_dir=output_dir + "/logs")
        '''
        This code is setting up a Trainer object that will manage the training loop, 
        including iterating through the training dataset, updating the model's weights, 
        and evaluating the model on the evaluation dataset using the provided compute metrics function.
        '''
        trainer = Trainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
                          compute_metrics=compute_metrics)
        '''
        Initiates the training process using the trainer.train() method. 
        This method runs the training loop, iterating over the provided training dataset and updating the 
        model's weights based on the defined training settings (training_args). 
        It returns the best-performing run based on the evaluation results.
        '''
        best_run = trainer.train()

        # best performing model is saved here
        trainer.save_model(output_dir)

    def train_with_validation_to_determine_hyperparameters(self, schema_type,
                                                           output_dir='../saved_models/saved_models_with_validation'):
        # we put some output_dir here where the model will be saved
        output_dir = output_dir + '/' + schema_type
        # We check whether a file already exists, if it exists this function gets skipped
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('[INFO] Model already exists, skip training')
            return

        # We obtain the encodings for the train and dev data
        train_encodings, train_labels = self.encode(self.train_dev_split_training_data,
                                                    output_dir=output_dir + '/train',
                                                    schema_type=schema_type)
        dev_encodings, dev_labels = self.encode(self.train_dev_split_dev_data, output_dir=output_dir + '/dev',
                                                schema_type=schema_type)

        # Basically a custom object storing the encodings and labels
        train_dataset = HFDataset(train_encodings, train_labels)
        dev_dataset = HFDataset(dev_encodings, dev_labels)

        '''
        Simply takes all the predictions and compares them with the true labels and 
        then applies accuracy performance metric.
        '''
        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            # we have both model predictions and ground truth labels
            '''
            logits typically represent the raw outputs or scores produced by the model for each class.
            In this case, it's assumed to contain the output logits for each class from the
            sequence classification model.
            '''

            logits, labels = eval_pred

            # Basically we take the index of the maximum amongst the classes
            # These ofcourse, represent our predictions
            predictions = np.argmax(logits, axis=-1)
            # Now we simply compute acuracy
            return metric.compute(predictions=predictions, references=labels)

        '''
        Code for hyperparameter tuning.

        '''

        # we define a functions which will select hyperparameters in a grid_search type style
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "num_train_epochs": trial.suggest_categorical("num_train_epochs",
                                                              [1, 2, 3, 4])
            }

        def model_init(trial):
            return AutoModelForSequenceClassification.from_pretrained(
                # I think I fixed this first one
                self.model_name,

                num_labels=2

                # I added this one, because we do binary classifation

                # This one is about whether the modelweights are from tensorflor. Set to False
                # from_tf=bool(".ckpt" in model_args.model_name_or_path),

                # this one is optional. I commented it out. We use standard config just like usual code.
                # config=config,

                # this one is optional. So we use standard cache_dir just like TIARA code.
                # cache_dir=model_args.cache_dir,

                # the following is a certain version or commit of the model. We can comment it out for now.
                # revision=model_args.model_revision,

                # i don't believe we need an authentification token.
                # token=True if model_args.use_auth_token else None,
            )

        training_args = TrainingArguments(do_train=True, do_eval=True,
                                          per_device_train_batch_size=256, per_device_eval_batch_size=64,
                                          output_dir=output_dir)

        trainer = Trainer(
            model=None,

            # these are the training arguments tot are not in the optuna search space or model_init.
            args=training_args,

            train_dataset=train_dataset,
            eval_dataset=dev_dataset,

            compute_metrics=compute_metrics,

            # this one is optional. Just like the TIARA code we do not include it.
            # tokenizer=tokenizer,

            model_init=model_init,

            # optional, just like TIARA code we do not include it.
            # data_collator=data_collator,
        )

        def compute_obj(dict):
            return dict['eval_accuracy']

        best_trials = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=1,
            compute_objective=compute_obj,
        )

        '''
        Now we take the optimal hyperparameters, and retrain the model with all the data we have for production.
        '''
        best_learning_rate = best_trials.hyperparameters['learning_rate']
        best_number_of_epochs = best_trials.hyperparameters['num_train_epochs']

        # CHANGE OUTPUT DIRECTORY
        output_dir = '../saved_models/saved_models_production' + '/' + schema_type
        train_encodings, train_labels = self.encode(self.production_training_data, output_dir=output_dir + '/train',
                                                    schema_type=schema_type)
        train_dataset = HFDataset(train_encodings, train_labels)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.train()

        training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_predict=True,
                                          per_device_train_batch_size=256, per_device_eval_batch_size=64,
                                          num_train_epochs=best_number_of_epochs, learning_rate=best_learning_rate)

        '''
        This code is setting up a Trainer object that will manage the training loop, 
        including iterating through the training dataset, updating the model's weights, 
        and evaluating the model on the evaluation dataset using the provided compute metrics function.
        '''
        trainer = Trainer(self.model, args=training_args, train_dataset=train_dataset)
        '''
        Initiates the training process using the trainer.train() method. 
        This method runs the training loop, iterating over the provided training dataset and updating the 
        model's weights based on the defined training settings (training_args). 
        It returns the best-performing run based on the evaluation results.
        '''
        trainer.train()

        # best performing model is saved here
        trainer.save_model(output_dir)

    def temporary_training_function(self, schema_type):
        best_learning_rate = 5e-5
        best_number_of_epochs = 3

        # CHANGE OUTPUT DIRECTORY
        output_dir = '../saved_models/saved_models_production' + '/' + schema_type
        train_encodings, train_labels = self.encode(self.train_dev_split_training_data,
                                                    output_dir=output_dir + '/train',
                                                    schema_type=schema_type)
        train_dataset = HFDataset(train_encodings, train_labels)

        self.model = AutoModelForSequenceClassification.from_pretrained('GroNLP/bert-base-dutch-cased', num_labels=2)
        self.model.train()

        training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_predict=True,
                                          per_device_train_batch_size=256, per_device_eval_batch_size=64,
                                          num_train_epochs=best_number_of_epochs, learning_rate=best_learning_rate)

        '''
        This code is setting up a Trainer object that will manage the training loop, 
        including iterating through the training dataset, updating the model's weights, 
        and evaluating the model on the evaluation dataset using the provided compute metrics function.
        '''
        trainer = Trainer(self.model, args=training_args, train_dataset=train_dataset)
        '''
        Initiates the training process using the trainer.train() method. 
        This method runs the training loop, iterating over the provided training dataset and updating the 
        model's weights based on the defined training settings (training_args). 
        It returns the best-performing run based on the evaluation results.
        '''
        trainer.train()

        # best performing model is saved here
        trainer.save_model(output_dir)

    def predict_question(self, question, schema_type, retrieve_top_k_schema_items,
                         dir="../saved_models/saved_models_production"):
        '''
        This function takes a question in natural language and uses a BERT dense retriever model in order to
        retrieve candidate schema items that could be relevant for the natural language question at hand.
        Based on the model inputs, it can either retrieve entity types (classes) or relations (properties).
        These are the main schema items present in knowledge graphs.
        This function retrieves the most likely schema items.

        :param question: The natural language questions asked by the user
        :param schema_type: Can be either 'class' or 'relation'.
        Depending on whether we want to retrieve candidate classes or candidate relations
        :param retrieve_top_k_schema_items: the amount of most likely schema items that mus tbe returned
        :param dir: This is part of the path where the BERT dense retriever model is saved.
        The path dynamically gets finished in the script based on the schema_type given as input to the function.

        :return: returns a list with the 'retrieve_top_k_schema_items' most likely schema items that the natural
         language questions is referencing.
        '''
        # we finish the directory in order to select the correct saved fine-tuned dense retriever neural network
        model_dir = dir + '/' + schema_type

        '''
        In Python, attributes can be created or modified outside of the class constructor 
        within any method or function of the class. 
        As long as the method or function setting the attribute is invoked after the class instantiation, 
        it will update or create the attribute as intended. That is why self.device wasn't explicitly mentioned
        in the class constructor.

        This line of code uses pytorch to set the device where the neural network operations will be performed.

        In this line of code a Pytorch device object is created for running operations on a CUDA-enabled GPU. (CUDA
        is a parallel computing  platform and application programming interface created by NVIDEA for their GPUs.
        This allows for faster computation in deep learning tasks compared to traditional CPUs). This is only done
        if a CUDA-enabled GPU is available. Otherwise it is simply set to CPU.
        '''
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # sequence classification model
        '''
        Note that the self.model attribute did not exist before.

        Sequence classification is a type of natural language processing (NLP) task
        where the goal is to classify a sequence of tokens (words, characters, etc.) 
        into predefined categories or classes. In this context, a "sequence" typically refers to a sentence, 
        a document, or any sequence of text.

        Note that AutoModelForSequenceClassification is part of Hugging Face's transformer library. It is a class
        used to automatically load an initialize a pre-trained model specifically designed for sequence classification
        tasks.

        The function .from_pretrained(output_dir) belonging to that class makes sure that we create an instance of the
        class by loading the model's weights and configuration from the directory specified by model_dir.
        This directory typically contains the pre-trained model's configuration files and trained weights, 
        allowing you to load a pre-trained model that was previously fine-tuned 
        or trained on a sequence classification task.

        Next, we set te model to evaluation mode. In Pytorch, if the .eval() method is called on a model, it prepares
        the model for evaluation of inference by disabling options such as dropout (regularization)
        and batch normalization. This ensures consistent behavior during predictions.

        The final line moves the model to the specified device where the computations will be performed
        '''
        # Note that output_dir should refer to a folder which contains:
        # 1) config.json 2) pytorch_model.bin 3) training_args.bin
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(self.device)

        '''
        First note that we use the following data for the ontology (can find this is constructor):
        self.all_class = read_list_file('../dataset/grail_classes.txt')
        self.all_relation = read_list_file('../dataset/grail_relations.txt')

        The purpose of the following piece of code is to divide a list of classes or relations into smaller chunks
        for processing.

        Based on the schema_type variable we take chunks of either classes or relations.

        chunk_size determines the size of the chunks. We simply use list comprehension to get the desired results.

        The 'assert' command double checks that ensures our final list is not empty, making sure at least
        one chunk has been created.

        The purpose of creating these chunks to process the data in pieces is related to memory efficiency 
        and computational performance. Note that processing data in chunks allows for parallel or concurrent 
        operations. 
        '''
        chunk_size = 500
        if schema_type == 'class':
            all_schema_chunks = [self.all_class[i:i + chunk_size] for i in range(0, len(self.all_class), chunk_size)]
        else:
            all_schema_chunks = [self.all_relation[i:i + chunk_size] for i in
                                 range(0, len(self.all_relation), chunk_size)]
        assert len(all_schema_chunks)

        # We create a list containing scores
        scores = []
        # We iterate through all the chunks
        for chunk_idx in range(0, len(all_schema_chunks)):
            try:
                # This returns the second column of the scores generated by the model
                # (this is the probability that the questions references a schema item)
                score = self.classify_question_schema(question, all_schema_chunks[chunk_idx])
                # We append the scores of a given chunk to the list which should contain the entire schema
                scores += list(score)
            except Exception as e:
                print('exception' + str(e))

        scores = np.array(scores)
        # Here we retrieve the indices of the top 100 scores
        predicted_idx = (-scores).argsort()[:retrieve_top_k_schema_items]

        # Then based on the indices we retrieve the actual classes or relations and put them in a list
        if schema_type == 'class':
            predicted_schema = [self.all_class[i] for i in predicted_idx[:retrieve_top_k_schema_items]]
        else:
            predicted_schema = [self.all_relation[i] for i in predicted_idx[:retrieve_top_k_schema_items]]

        return predicted_schema

    '''
    This function simply can start classifying a certain question given a chunk of schema items.
    We run this function on a question with all the schema chunks in order to get our final results.


    '''

    def classify_question_schema(self, question: str, schema: list):
        # We create a list containing the question repeated as many times as the length of schema
        text_a = [question for _ in range(0, len(schema))]
        # this will simply be the schema
        text_b = schema
        try:
            # We tokenize both text_a and text_b and we specify that the function must return pytorch tensors
            encodings = self.tokenizer(text_a, text_b, max_length=128, truncation=True, padding=True,
                                       return_tensors='pt')

            # Note that this classify_question_schema() function is used in the predict() function.
            # In the predict function we set self.model to be specified for evaluation.
            '''
            Note that the encodings object contains
            1) input_ids
            This represents the tokenized input text converted into numerical indices.
            corresponding to tokens in the model's vocabulary. Each token in the input text is mapped to its own unique
            index in the models vocabulary. It's basically a list or tensor of integers where each integer represents 
            the index of a token in the tokenizer's vocabulary.

            NOTE: these are not the same as embeddings. It simply represents the tokenized input converted into 
            numerical indices which refer to tokens in model's vocabulary. These IDs are inputs to a model during
            inference or training. These input_ids get transformed into embeddings when fed into the model.

            2)attention_mask
            This is a binary tensor indicating which tokens should be attended to and which ones should be ignored
            by the model during processing. It has the same length as input_ids. In the attention mask we simply have 
            a 1 or a 0 indicating which tokens should be processed or ignored.

            This attention mask is needed because in the context of using mini-batches in deep learning, 
            sequences often need to be of the same length for efficient computation. Especially for recurrent
            or transformer-based models. To ensure this padding tokens are added to the input sequences to make them
            all have the same length as the longest sequence in the batch.

            The attention mask helps the model distinguish between actual tokens and padded tokens.

            Note that the attention_mask is seperate from the core attention mechanism but is essential for how
            the attention mechanism operates within the model. The core attention mechanism assigns weights to
            each token based on its relevancy to other tokens. The attention_mask ensures that padding tokens
            get assigned a weight of 0.

            3)token_type_ids
            This one is used in models accept multiple sequences as input. (for example in question answering
            or in sentence-pair classification)

            In question-answering it differentiates between tokens from the question and tokens from the answer.
            It's a list or tensor of integers, the same length as input_id, where each token is assigned an type id
            based on its segment of sequence.

            '''

            # Here we simply get the prediction by passing the encodings into the model that was created before.
            predictions = self.model(input_ids=encodings['input_ids'].to(self.device),
                                     attention_mask=encodings['attention_mask'].to(self.device),
                                     token_type_ids=encodings['token_type_ids'].to(self.device))

            # Here we retrieve logits (raw scores) and convert it into a numpy array
            scores = predictions.logits.detach().cpu().numpy()

            # We only return the values of the second column
            return scores[:, 1]
        except Exception as e:
            print('classify_question_schema: ' + str(e))


if __name__ == '__main__':
    # First we look in the command line what specification are chosen
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--schema_type', choices=['class', 'relation'], required=True)
    # parser.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    # simply an object that holds the arguments:
    # args = parser.parse_args()

    # We create a GrailQASchemaDenseRetriever object
    # schema_dense_retriever = KadasterSchemaDenseRetriever()
    # We fetch input that is given in the terminal.
    # We train the model here (note the train function checks whether the model is already trained)
    # schema_dense_retriever.train(args.schema_type)
    # split = None
    # Based on the input in the terminal we select the appropriate dataloader.

    '''
    Note the following attributes that are set in the script based on the path.
    We can look at the GrailQAJsonLoader object to see how everything works.
    self.grailqa_train = GrailQAJsonLoader(grailqa_train_path)
    self.grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)
    self.grailqa_test = GrailQAJsonLoader(grailqa_test_path)
    '''

    # if args.split == 'train':
    #    split = schema_dense_retriever.grailqa_train
    # elif args.split == 'dev':
    #    split = schema_dense_retriever.grailqa_dev
    # elif args.split == 'test':
    #    split = schema_dense_retriever.grailqa_test
    # schema_dense_retriever.predict(split, '../model/schema_dense_retriever', args.schema_type)

    import os

    os.environ['CURL_CA_BUNDLE'] = ''

    KKG_schema_retriever = KadasterSchemaDenseRetriever()
    # KKG_schema_retriever.train_with_validation_to_determine_hyperparameters('class',output_dir='../saved_models/saved_models_with_validation')
    # KKG_schema_retriever.temporary_training_function('relation')
    print(KKG_schema_retriever.predict_question('hoeveel arubanen zijn er in apeldoorn?', 'relation', 20))

