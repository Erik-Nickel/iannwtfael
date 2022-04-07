import training


def print_commands():
    print("Hey this is a food recommender app!")
    print("You can insert your favourite recipes and get a recommendation based on them :)")
    print("Use the following commands to navigate the app:")
    print("exit: closes the app")
    print("list: lists all recipes with their id and name")
    print("rec: gives a recommendation bast on your preferences")


def list_recipes():
    pass


def recipe_data_from_ids(ids):
    pass


def string_to_list(inp):
    pass


def pred(ids):
    model = training.create_model(seq_len=9)
    training.load_wights(model)
    data = None
    prediction = model.prediction(data)
    recipe_details(prediction)


def recipe_details(id):
    pass


if __name__ == "__main__":
    while True:
        print_commands()
        command = input()
        if command.__eq__('exit'):
            break
        elif command.__eq__('list'):
            list_recipes()
        elif command.__eq__('rec'):
            id = None
            recipe_details(id)
        elif command.__eq__('pred'):
            print("insert 9 liked recipes (recipe IDs) as list (e.g: [7, 99, 48, 1000, 4, 9999]): ")
            ids = input()
            pred(string_to_list(ids))
