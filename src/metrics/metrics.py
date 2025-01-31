import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Cargar los archivos CSV
human_texts_df = pd.read_csv('mano_mix.csv')
generated_texts_df = pd.read_csv('full_mix.csv')

# Convertir los DataFrames a diccionarios para facilitar la búsqueda
human_texts_dict = dict(zip(human_texts_df['image_name'], human_texts_df['caption']))
generated_texts_dict = dict(zip(generated_texts_df['image_name'], generated_texts_df['best_description_spanish']))

# Encontrar las imágenes comunes en ambos archivos
common_images = set(human_texts_dict.keys()).intersection(set(generated_texts_dict.keys()))

print("Cantidad de imágenes en común: ", +len(common_images))

# Extraer los textos correspondientes a las imágenes comunes
human_texts = [human_texts_dict[image] for image in common_images]
generated_texts = [generated_texts_dict[image] for image in common_images]


# Función para calcular BLEU
def calculate_bleu(human_text, generated_text):
    reference = [human_text.split()]
    candidate = generated_text.split()
    return sentence_bleu(reference, candidate)


# Función para calcular METEOR
def calculate_meteor(human_text, generated_text):
    return meteor_score([human_text.split()], generated_text.split())


# Función para calcular ROUGE
def calculate_rouge(human_text, generated_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(human_text, generated_text)
    return scores


# Calcular métricas para cada par de textos
bleu_scores = [calculate_bleu(human, generated) for human, generated in zip(human_texts, generated_texts)]
meteor_scores = [calculate_meteor(human, generated) for human, generated in zip(human_texts, generated_texts)]
rouge_scores = [calculate_rouge(human, generated) for human, generated in zip(human_texts, generated_texts)]

# Calcular promedios de ROUGE
average_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0
average_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0

# Calcular promedios de BLEU y METEOR
average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
average_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0


gts = {img_id: [human_texts_dict[img_id]] for img_id in common_images}
res = {img_id: [generated_texts_dict[img_id]] for img_id in common_images}

cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(gts, res)
print(f"CIDEr score: {cider_score}")

spice_scorer = Spice()
spice_score, _ = spice_scorer.compute_score(gts, res)
print(f"SPICE score: {spice_score}")


# Imprimir resultados
print(f"Average BLEU score: {average_bleu}")
print(f"Average METEOR score: {average_meteor}")
print(f"Average ROUGE-1 F1 score: {average_rouge1}")
print(f"Average ROUGE-L F1 score: {average_rougeL}")