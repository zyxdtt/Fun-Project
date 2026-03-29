# RNN_AI-bot: A Weird but Surprisingly Grammatical Chatbot 🤖

This is my first deep learning project as a freshman. I built a generative dialogue AI from scratch using an RNN (LSTM). The result is a bot that has perfect English grammar but absolutely zero understanding of the world. It will apologize endlessly and is obsessed with the word "seat" and the symbol "?".

**It’s dumb, but it’s mine.**

## Dataset 📚

I used the **DailyDialog** dataset for training. It contains 13,118 multi-turn dialogues (about 8 rounds per dialogue) on everyday topics.

- Data source: [OpenDataLab - DailyDialog](https://opendatalab.org.cn/OpenDataLab/DailyDialog/tree/main)
- File used: `dialogues_text.txt`

## Project Files 📁

| File | Description |
|------|-------------|
| `preprocessing.cpp` | C++ code for data preprocessing. Builds vocabulary (10,000 words) and converts dialogue texts into ID sequences. Runs in milliseconds. |
| `RNN_train.py` | Python script using TensorFlow/Keras. Defines and trains the LSTM model (Embedding 128, LSTM 256). Saves the best model as `best_model.h5`. |
| `chat_test.py` | Interactive script to chat with the trained model. Converts your input to IDs, feeds to the model, and prints the generated response word by word. |

## Trained Model 🤗

The trained model file `best_model.h5` is available on Hugging Face:

👉 [zyxdtt/RNN_AI-bot on Hugging Face](https://huggingface.co/zyxdtt/RNN_AI-bot/tree/main)

## How Dumb Is It? 🤪

See the chat screenshot in this repo. Here's a typical conversation:
User: How are you?
Bot: I am a student in the mood for a while .
User: What are you studying?
Bot: I am applying for a vacation .
User: Did you get the vacation?
Bot: Yes , I am going to get a lot of the ? .
User: Where will you go?
Bot: I don't know what's a ? .
User: Seat?
Bot: Yes , I am going to get a lot of the ? .
User: ?
Bot: I don't know what's a seat .
### The Bot's Personality 🧠

- **Grammar**: Perfect. It uses correct tenses, prepositions, and punctuation.
- **Politeness**: Unconditional. It will apologize for anything (`I am sorry.`).
- **Knowledge**: Knows three things: "seat", "?", and how to say sorry.
- **Logic**: Non-existent. But it never fails to be entertaining.

**It's like a student who memorized all the grammar rules but never learned what any word actually means.**

## Why I'm Proud of This 💪

- Built a complete AI pipeline: C++ preprocessing → Python training → interactive chat
- Trained a generative dialogue model from scratch (no pretrained embeddings)
- The model **learned** to produce grammatically correct English, even if it doesn't understand semantics

## Limitations 🚧

- Small dataset and short training (10 epochs, 50 context window)
- No GPU, so it's slow
- Frequently gets stuck in loops ("I am sorry.")
- Obsessed with "seat" and "?"

## Future Plans 🔮

- [ ] Train longer (more epochs)
- [ ] Use pretrained embeddings (GloVe)
- [ ] Try beam search for better generation
- [ ] Maybe teach it what a "seat" actually is

## Credits

Built from scratch for learning purposes. Inspired by the desire to understand how ChatGPT works (at a very small scale).

---

**This bot is my first step into AI. It's dumb, but it's mine.** 🚀
