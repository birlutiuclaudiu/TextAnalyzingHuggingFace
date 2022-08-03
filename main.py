import torch.cuda
import torch.nn.functional as TF
import torch as TORCH
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer  # https://www.sbert.net/
from transformers import MBartForConditionalGeneration as TranslateModel, MBartTokenizer as TranslateTokenizer
from transformers import DistilBertTokenizerFast as DBTF
from sklearn.metrics.pairwise import cosine_similarity
# for data training
import os as OS
import requests
import json as JSON  # pentru citirea datelor din fisiere de tip json
from transformers import DistilBertForQuestionAnswering as DBFQA, BertForQuestionAnswering as bfqa
from torch.utils.data import DataLoader
from transformers import AdamW as ADAM
from tqdm import tqdm


def util_print_sep():
    print("-----------------------------------------------------------------------------------------------------------")


""" -------------------------------------------CREAREA UNUI SMART HOME-----------------------------------------------"""
""" In acest context, in functie de actiunea sau perioada zilei in care suntem, programul va alege o actiune cat mai fezabila 
de facut din setul de 8 propozitii"""

sentences = [
    "Turn on the light in bedroom",
    "Turn on the light in kitchen",
    "Turn on all lights because is night",
    "Turn off all lights because is day",
    "Turn on light in the kitchen",
    "Turn on the light in the study room",
    "Save water!",
    "Turn off the light in house",
]


def choose_best_option(input_txt):
    """Aceasta metoda determina similititudini intre propozitia primita ca input si cele 6 porpozitii definite mai sus"""
    # definirea tokenizatorului
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v1")
    # propozitiile vor fi encodate folosind metoda encode
    embeddings = model.encode(sentences)
    # encodarea propozitiei/textului primit ca input
    input_encode = model.encode(input_txt)
    util_print_sep()
    print(f"This is the encoded value for the input: \n    {input_encode}")
    # Print the embeddings
    print("This is the encoded value for the option sentences")
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        util_print_sep()
    # pentru a vedea similitudinea dintre propozitia input si propozitiile prezente ca optiuni se va folosi functia
    # cosine_similarity
    similarity_values = cosine_similarity([input_encode], embeddings[0:])
    # se va determina propozaitia cea mai apropiata de cea data ca input; se va alege propozitia cu scorul de matching cel
    # mai mare
    max_value = max(similarity_values[0])
    print(f"Similarity rates: {similarity_values}")
    util_print_sep()
    print(f"Max value: {max_value}")
    max_index = [index for index, item in enumerate(similarity_values[0]) if item == max_value]
    to_return_value = sentences[max_index[0]] + " (with a score " + str(max_value) + ")"
    print(f"The result\n    {to_return_value}")
    return to_return_value


"""-----------------------------------------SENTIMENTAL ANALYSIS ------------------------------------------------------"""
DIR_DATASET_SENTIMENTAL = "dataset_text_classification"


def text_classification(train):
    """se va crea modelul și functia de tokenizare din directorul "dataset_text_classification"
    acest director contine data_set-ul "distilbert-base-uncased-finetuned-sst-2-english", model
    ce atinge o acuratete de 91.1 pe dev set"""

    print(f"Pretrained model instantiate from:  {DIR_DATASET_SENTIMENTAL}")
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(DIR_DATASET_SENTIMENTAL)
    # tokenizatorul va fi preluat din modelul gata antrenat
    util_print_sep()
    print("Tokenizer imported")
    tokenizer = AutoTokenizer.from_pretrained(DIR_DATASET_SENTIMENTAL)

    """pentru a obtibne tokenii textului dat ca input se va folosi urmatoarea metoda; in principiu fiecare cuvant din
    pozitie e un token; deci fiecarui token ii este atasat un id unic si aceasta reflecta reprezentarea matematica pe care
    modelul nostru o va recunoaste/intelege"""

    available_tokens = tokenizer.tokenize(train)
    util_print_sep()
    print(f"Available tokens form input\n    {available_tokens}")
    # se converteste fiecare token la un id
    token_map_to_ids = tokenizer.convert_tokens_to_ids(available_tokens)
    util_print_sep()
    print(f"Tokens map to ids: {token_map_to_ids}")
    # in continuare se va defini clasificatorul
    model_classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer)
    # se va apela clasificatorul pe ficare line de text; analizand daca numarul de
    output_data = model_classifier(train.split('\n'))
    # se formeaza rezultatul de afisat in interfata grafica
    output_to_print = ""
    index = 0
    for data in output_data:
        index += 1
        output_to_print += (
                "LINE " + str(index) + ":" + data['label'] + " with the score " + str(data['score']) + "\n")
    print(output_to_print)
    util_print_sep()
    return output_to_print


"""--------------------------------------IMPORTAREA MODELULUI DE TRADUCERE EN-RO-------------------------------------"""


def translte_en_ro(input):
    """Aceasta metoda se va ocupa de traducerea tecxtului introdus de utilizator din englea in romana"""
    """see: https://huggingface.co/docs/transformers/model_doc/mbart """
    # definirea modelului dintr-un model antrenat
    util_print_sep()
    print("Import model from a pretrained model: facebook/mbart-large-en-ro")
    model = TranslateModel.from_pretrained("facebook/mbart-large-en-ro")
    # initializarea tokenizatorului
    print("Instantiated tokenizer from: facebook/mbart-large-en-ro")
    tokenizer = TranslateTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")
    # obtinerea batch-ului din tokenizarea textului de input
    batch = tokenizer(input, return_tensors="pt")
    extracted_tokens_translated = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
    # se vaor decodifica tokenii obtinuti
    return tokenizer.batch_decode(extracted_tokens_translated, skip_special_tokens=True)[0]


"""----------------------------------Crearea propriului model cu date preluate din fisier json-----------------------"""
DIR_DATASET_TRAINING = 'dataset_training'  # in acest folder se vor salva cele doua fisere json aduse de la url-ul dat
TRAINING_FILE = 'train-v2.0.json'  # denumirea fisierului
DEV_FILE = 'dev-v2.0.json'
url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
# acest numar reprezinta numarul de date pe care se vor excecuta epocile; problema pentru care se foloseste un numar atat
# de mic e performanta calculatorului
NUMBER_OF_DATA = 20


def write_in_file(file_path):
    """Aceasta metoda preia de la url dataset-ul sub forma json si il salveaza local in fisierul de la locatia
     data de stringul primit ca parametru file_path """
    util_print_sep()
    print(f"Writing the value extracted from url in file : {file_path}")
    training_set = requests.get(f"{url}/{file_path}",
                                stream=True)  # se face cererea spre preluarea datelor de training
    with open(f"{DIR_DATASET_TRAINING}/{file_path}", 'wb') as file:  # se salveaza datele de training intr-un fisier
        util_print_sep()
        print(f"Writng the data using chunk with size of 4")
        for buffer in training_set.iter_content(chunk_size=4):
            file.write(buffer)


def import_dataset_from_url():
    """se vor lua 2 fisiere de la url-ul respectiv respectiv"""
    util_print_sep()
    print(f"Import {TRAINING_FILE}")
    write_in_file(TRAINING_FILE)     # se preiau datele de training
    util_print_sep()
    print(f"Import {DEV_FILE}")
    write_in_file(DEV_FILE)          # se preiau datele de dev


def read_json_file(file_path):
    """se incarcca fisieurl de tip JSOn in memorie si dupa procesere se va returna un dicitionar cu campurile\
    pe care le contine training_dataset reprezinta un dicitionar ce contine date grupate in groups/data, passaje
    sau paragrafe in interiorulu grupurilor unde se regaseste contextul pasajului si intrebarile corespunzatoare"""

    util_print_sep()
    print(f"Open the file JSON: {file_path}")
    #se deschide fisierul al carui path e dat ca parametru in modul citire
    with open(f"{DIR_DATASET_TRAINING}/{file_path}", 'rb') as file:
        training_dataset = JSON.load(file)

    #s-a definit un dictionar care cuprinde cele 3 tipuri de date ce se regasesc in fisierul JSON

    dictionary = {'contexts': [], 'questions': [], 'answers': []}
    util_print_sep()
    print(f"Create empty dictionary to upload data with folowing fileds: {dictionary.keys()}")
    #Se vor extrage campurile relevante pentru training si respectiv dev"""
    for i in range(0, len(training_dataset['data'])):
        #pentru o data avem un context(paragraph) care are un set de intrebari si la randul lor au un set de raspunsuri
        data = training_dataset['data'][i]
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            #pentru fiecare paragraf sau context se va naviga prin setul de intrabri asociat acestuia
            for question_answer in paragraph['qas']:
                question = question_answer['question']

                #se verifica daca sunt raspunsuri plauzibile pentru intrebaare in setul de keys ale acesteia
                field = 'plausible_answers' if 'plausible_answers' in question_answer.keys() else 'answers'

                # se adauga in dictionar datele corespunszatoare contextului cu intrebarea si raspunsul ei specifice
                for ans in question_answer[field]:
                    dictionary['answers'].append(ans)
                    dictionary['contexts'].append(context)
                    dictionary['questions'].append(question)
    return dictionary


def update_answers_with_end_index(dictionary):
    """Deoarece dictionarul e de forma {'text': 'in the late 1990s', 'answer_start': 269}, vrem sa aflam si indexul de final si
     sa il adaugam raspunsului. Aceasta metodda se va ocupa de problema descrisa"""

    util_print_sep()
    print(f"Update answer filed from dictionary : {dictionary.keys()} with end_index field")
    # se determina pentru fiecare raspuns indexul de final
    for answer, context in zip(dictionary['answers'], dictionary['contexts']):
        #indexul de end este dat de indexul initial + lungimea textului raspunsului
        answer_end = answer['answer_start'] + len(answer['text'])
        if context[answer['answer_start']:answer_end] != answer['text']:
            # daca in context nu avem raspunsul se vor actualiza punctele de start si end pentru context
            for i in range(1, 3):
                if context[answer['answer_start'] - i:answer_end - i] == answer['text']:
                    answer['answer_start'] = answer['answer_start'] - i
                    answer['answer_end'] = answer_end - i
        else:
            # se va primi chiar answer end
            answer['answer_end'] = answer_end


def positioned_tokens(encode_dict, training_cqa, tokenizer):
    """Aceasta metoda se va ocupa cu definirea pozitiei fiecarui token din cele doua dictionare"""
    # update campuri encoding cu start posittion si end position
    encode_dict.update({'positions_start': [], 'positions_end': []})
    # se vor adauga pozitiile de start si de end pentru cele 100000 de tokeni ce au putut fi generati;  capacitatea cal-
    # calatorului nu a permis rularea a mai mult de 100000 de date

    for n in range(NUMBER_OF_DATA):
        to_write = 1
        encode_dict['positions_start'].append(encode_dict.char_to_token(n, training_cqa['answers'][n]['answer_start']))
        encode_dict['positions_end'].append(encode_dict.char_to_token(n, training_cqa['answers'][n]['answer_end']))
        if encode_dict['positions_start'][-1] is None:
            encode_dict['positions_start'][-1] = tokenizer.model_max_length
        while encode_dict['positions_end'][-1] is None:  # daca ultimul element nu exista, atumci se duplica ultimul
            encode_dict['positions_end'][-1] = encode_dict.char_to_token(n,
                                                                         training_cqa['answers'][n][
                                                                             'answer_end'] - to_write)
            to_write = to_write + 1


class TorchDataSet(TORCH.utils.data.Dataset):
    """Aceasta clasa reprezinta o modalitate de a transfoma datele codificate intr-un obiect de tip pytorch
    transformarea dataset-urilor intr-un astfel de obiect va permite procesul de antrenarea a modelului pe baza datelor
    existente """
    #https://www.youtube.com/watch?v=ZIRmXkHp0-c&t=2405s
    def __init__(self, encode_data):
        self.encode_data = encode_data

    def __len__(self):
        return len(self.encode_data.input_ids)

    def __getitem__(self, item):
        return {key: TORCH.tensor(val[item]) for key, val in self.encode_data.items()}

    def get_name(self):
        return self.encode_data


def save_the_model_in_device(model, tokenizer):
    util_print_sep()
    print(f"Save the model and tokenizer in : {'new_qa_model/my_distilbert'}")
    model_path = 'new_qa_model/my_distilbert'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def training_data(input_text):
    """Aceasta e metoda principala care descrie flow-ul de creare a propriului model"""
    # se vor aduce datele de test la url dat doar daca nu exista deja
    util_print_sep()
    print(f"Verify existence of the folder with the datasets from {DIR_DATASET_TRAINING}")
    if not OS.path.exists(DIR_DATASET_TRAINING):  # daca fisierul deja exista, atunci nu se va mai crea si aduce datele
        OS.makedirs(DIR_DATASET_TRAINING)  # se creeaza folderul in care ser vor salva cel e doua fisiere
        util_print_sep()
        print(f"Create the folder:  {DIR_DATASET_TRAINING} and import data from the url path")
        import_dataset_from_url()

    # se vor extrage circumstantele posibile cu intrebrile si raspunsurile pentru partea de training
    util_print_sep()
    print(f"Load training dataset from JSON file")
    training_cqa = read_json_file(TRAINING_FILE)
    print(f"Load dev dataset from JSON file")
    # se vor extrage circumstantele posibile cu intrebarile si raspunsurile pentru partea de dev
    dev_cqa = read_json_file(DEV_FILE)

    util_print_sep()
    print(f"The first answer without any modification from training {training_cqa['answers'][0]}")
    util_print_sep()
    print(f"The first answer without any modification from dev {dev_cqa['answers'][0]}")
    util_print_sep()
    print(f"The first question without any modification from training {training_cqa['questions'][0]}")
    util_print_sep()
    print(f"The first question without any modification from dev {training_cqa['questions'][0]}")

    # se vor actualiza raspunsurile cu indexxul lor de final pentru cele doua dataset-uriL training si dev
    util_print_sep()
    print(f"Update the answers from training dataset and dev dataset with the end index")
    update_answers_with_end_index(training_cqa)
    update_answers_with_end_index(dev_cqa)
    util_print_sep()
    print(
        f"A listing of the first two answers after update the start points and end points{training_cqa['answers'][:2]}")


    # URMATORUL PAS ESTE codificarea datelor obtinute si obtinerea de tokeni
    # se defineste tokenizerul - acesta se foloseste de transformerul DistilBertTokenizerFast; se foloses

    tokenizer = DBTF.from_pretrained('distilbert-base-uncased')  # definit pe modelul antrenat 'distilbert-base-uncased'
    util_print_sep()
    print(f"Instantiate tokenizer from distilbert-base-uncased ")

    # se vor codifica valorile; prin merge-uirea celor 2 stringuri si rezultrarea a 2 seturi de tokens : pentru
    # context si questions am ales sa se faca tokenizare doar pe NUMBER_OF_DATA de date din cauza faptului ca se va
    # ajunge la exces de memorie l a o rulare pe un dataset mare
    encode_training_dataset = tokenizer(training_cqa['contexts'][:NUMBER_OF_DATA],
                                        training_cqa['questions'][:NUMBER_OF_DATA],
                                        padding=True,
                                        truncation=True)
    encode_dev_dataset = tokenizer(dev_cqa['contexts'][:NUMBER_OF_DATA], dev_cqa['questions'][:NUMBER_OF_DATA],
                                   padding=True,
                                   truncation=True)
    # s-au convertit astfel valorile anterioare la niste obiecte codificate
    util_print_sep()
    print(f"Token dictionary format: {encode_training_dataset.keys()}")
    util_print_sep()
    print(f"Input ids of the first token from training set: {encode_training_dataset['input_ids'][0]}")
    util_print_sep()
    print(f"Input ids of the first token from dev set: {encode_dev_dataset['input_ids'][0]}")
    # in urma tokenizarii se va obtine un dictionar de forma dict_keys(['input_ids', 'attention_mask'])
    
    # ADAUGRAREA POZITIEI TOKENILOR
    # adaugarea pentru training encode
    positioned_tokens(encode_training_dataset, training_cqa, tokenizer)
    util_print_sep()
    print(f"First token of training set after added positions: \n   {encode_training_dataset['input_ids'][0]}\n   "
          f"{encode_training_dataset['attention_mask'][0]}\n  {encode_training_dataset['positions_start'][0]}\n   "
          f"{encode_training_dataset['positions_end'][0]}\n ")
    # adaugarea celor doua pozitii pentru dev encode
    positioned_tokens(encode_dev_dataset, dev_cqa, tokenizer)
    util_print_sep()
    print(f"First token of training set after added positions: \n   {encode_dev_dataset['input_ids'][0]}\n   "
          f"{encode_dev_dataset['attention_mask'][0]}\n  {encode_dev_dataset['positions_start'][0]}\n   "
          f"{encode_dev_dataset['positions_end'][0]}\n ")

    # in continuare se va crea modelul antrenat. beneficiile de a folosi un model preantrenat este ca va reduce costurile de
    # computatie
    # PARTEA URMATOAREA SE VA OCUPA DE FINE TUNE
    model = DBFQA.from_pretrained('distilbert-base-uncased')  # acesta este un model deja antrenat de care ne vom folosi
    # pythorc fa volosi fie cpu or gpu; in urmatoarele linii de cod se determina cu ce parte a sistemului de calcul se va
    # realiza procearea datelor
    #definirea partii de
    util_print_sep()
    print(f"Choosed CPU to execute the epochs")
    cpu_device = torch.device('cpu')
    model.to(cpu_device)  # transferul modelului spre device pentru computatie
    model.train()  # antranrarea modelului
    optimizer = ADAM(model.parameters(), lr=5e-5)  # definireea ooptimizatorului
    # formarea celor doua obiecte de tip pytorch a datelor
    train_cqa_dataset = TorchDataSet(encode_training_dataset)
    dev_cqa_dataset = TorchDataSet(encode_dev_dataset)
    #se creeaza setul de training permitand amestecarea datelor
    util_print_sep()
    print(f"Created data loader for training set allowing to shuffle data")
    train_loader = DataLoader(train_cqa_dataset, batch_size=16,
                              shuffle=True)  # shuffle este activat pentru amestecarea datelor
    util_print_sep()
    print(f"Created data loader for dev set allowing to shuffle data")
    dev_loader = DataLoader(dev_cqa_dataset, batch_size=16,
                            shuffle=True)  # shuffle este activat pentru amestecarea datelor

    results = None
    # in continuare se vor rula 2 epoci pentru antrenare datasetului ACESTA E PROCESUL DE FINE-TUNE
    for epoch in [1, 2]:
        iterations = tqdm(train_loader)
        for batch in iterations:
            optimizer.zero_grad()
            #orocesarea de catre device datelor obitnure din obicetul de tip pytorch
            input_ids = batch['input_ids'].to(cpu_device)
            attention_mask = batch['attention_mask'].to(cpu_device)
            positions_start = batch['positions_start'].to(cpu_device)
            positions_end = batch['positions_end'].to(cpu_device)
            #se creeaza setul de rezultate
            results = model(input_ids, attention_mask=attention_mask, start_positions=positions_start,
                            end_positions=positions_end)
            #se exytrage loss-ul din rezultate
            loss = results[0]
            loss.backward()
            #se trece la urmatorul pas
            optimizer.step()
            #setarea indexului epocii
            iterations.set_description(f"The Epoch {epoch} execution: ")
            iterations.set_postfix(loss=loss.item())
    util_print_sep()
    print(f"On result executing the 2 epochs\n     {results[0]}")
    # salvarea modelului custom pe decvice
    save_the_model_in_device(model, tokenizer)
    results = None
    all_acc = []  # initializare
    ###Acum se va face pentru partea de dev;
    iteration = tqdm(dev_loader)
    for batch in iteration:
        with TORCH.no_grad():
            input_ids = batch['input_ids'].to(cpu_device)
            attention_mask = batch['attention_mask'].to(cpu_device)
            positions_start = batch['positions_start'].to(cpu_device)
            positions_end = batch['positions_end'].to(cpu_device)
            #extragerea rezultatelor sub forma de model cu input id-urile si attention mask-ul definit
            results = model(input_ids, attention_mask=attention_mask)
            #extragere positiilor de start si end
            start = TORCH.argmax(results['start_logits'], dim=1)
            end = TORCH.argmax(results['end_logits'], dim=1)
            #actualizarea datelor de acumulare peentru numarul
            all_acc.append(((start == positions_start).sum() / len(start)).item())
            all_acc.append(((end == positions_end).sum() / len(end)).item())
    util_print_sep()
    print(f"A form of output after process the dev loader:\n{results}")
    util_print_sep()
    print(f"The accuracy of the model: {sum(all_acc) / len(all_acc)}")
    #executarea unui model adevarat
    qa_model = pipeline("question-answering")
    question = input_text.split('\n')[1]  #prima linie reprezinta intrebarea
    context = input_text.split('\n')[0]   #a doa linie reprezinta contextul
    result = qa_model(question=question, context=context)
    print(result)
    return f"The accuracy of the custom model is:  {sum(all_acc) / len(all_acc)} because the training was executed on " \
           f"200 data. The problem was the memory of GPU or CPU which limited the execution of epochs" \
           f"\n\n The answer for the question is \n    {result}"


"""-----------------------------------------DEFINIREA INTERFETEI GRAFICE CU UTUILIZATORUL----------------------------"""


def analyzer(input, type):
    """ Aceasta metoda va fi cea executata in momentul in care este apasat butonul de submmit din interfata grafica.
    In functie de tipul de operatie pe care sa il execute, va apela una dintere metodele descrise mai sus ce prezinta un
    tip de anliza NLP"""
    if type == "sentimental_classifier":
        # va clasifica fiecare linie din input in functie de sentimentul ce reiese din aceasta
        util_print_sep()
        print("SENTIMENT CLASSIFIER")
        return text_classification(input)
    if type == "SMART_HOME_option_based_on_similarity":
        # va determina pentru actiunea descrisa in input cea mai buna optiune penteru smart home din cele existente
        util_print_sep()
        print("SMART HOME")
        return choose_best_option(input)
    if type == "TRANSLATION_EN_TO_RO":
        # va determina pentru propozitia scrisa in input in engleza, propozitia tradusa in romana
        util_print_sep()
        print("TRANSLATION EN RO")
        return translte_en_ro(input)
    if type == "CREATE_OWN_MODEL_QA":
        # se va crea propriu model din antrenarea unui set de date;
        util_print_sep()
        print("CREATE OWN MODEL")
        return training_data(input)
    return ""


def define_user_interface():
    """ Este definita o interfata grafica cu utilizatorul in care se poate alege o actiune pe care sa o faca sistemul
    pe inoutul dat"""
    demo = gr.Interface(
        analyzer,
        ["text", gr.Radio(
            ["sentimental_classifier", "SMART_HOME_option_based_on_similarity", "TRANSLATION_EN_TO_RO",
             "CREATE_OWN_MODEL_QA"])],
        outputs=["text"],
        examples=[
            [f"I am a good person.\nHe is a bad person", "sentimental_classifier", "positive or negative"],
            ["I want to cook something", "SMART_HOME_option_based_on_similarity", "one suggestion"],
            ["I'm going to school", "TRANSLATION_EN_TO_RO", "romaninan text"],
            ["I am a good person.\nHow I am?", "CREATE_OWN_MODEL_QA", ""],
        ],
        title="Birlutiu Claudiu-Andrei - LFT project",
        description="Here is a user interface which allows user to introduce data and view the analysis on that data ",
        flagging_options=["this", "or", "that"],
    )
    demo.launch()


if __name__ == '__main__':
    define_user_interface()
