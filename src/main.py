from household import Household

if __name__ == "__main__":
    h1 = Household(1)
    h1.build_classifier()
    h1.train(epochs=100)
    loss, accuracy = h1.evaluate()
    print(f"Accuracy: {accuracy}")


