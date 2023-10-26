from dataset.dataset import get_images_path_and_name

def main():
    train_img_df = get_images_path_and_name("./dataset/images/")
    test_img_df = get_images_path_and_name("./dataset/Test/images/")
    print(train_img_df)
    print(test_img_df)


if __name__ == '__main__':
    main()
