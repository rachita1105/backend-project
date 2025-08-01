from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import sys

def sentiment_analysis():
    print("\n📊 Sentiment Analysis Selected")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    text = input("\nEnter a sentence for sentiment analysis: ")
    result = sentiment_pipeline(text)
    print("Result:", result)


def summarization():
    print("\n📝 Text Summarization Selected")

    model_name = "facebook/bart-large-cnn"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    text = input("\nEnter a long paragraph to summarize:\n")
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    print("\nSummary:", result[0]['summary_text'])


def translation():
    print("\n🌐 English to French Translation Selected")

    model_name = "Helsinki-NLP/opus-mt-en-fr"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    translator = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)

    text = input("\nEnter English sentence to translate to French: ")
    result = translator(text, max_length=100)
    print("French Translation:", result[0]['translation_text'])


def main():
    print("\n🤖 AI CLI Assistant")
    print("Choose a task:")
    print("1. Sentiment Analysis")
    print("2. Text Summarization")
    print("3. Translation (English → French)")

    choice = input("\nEnter 1, 2, or 3: ")

    if choice == '1':
        sentiment_analysis()
    elif choice == '2':
        summarization()
    elif choice == '3':
        translation()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
