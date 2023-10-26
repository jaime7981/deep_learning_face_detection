from dataset.dataset import get_images_path_and_name, get_labels
from retinaface_copy.RetinaFace import build_model, extract_faces, detect_faces

def main():
    train_img_df = get_images_path_and_name("./dataset/images/")
    test_img_df = get_images_path_and_name("./dataset/Test/images/")

    labels = get_labels(train_img_df)
    print(train_img_df)
    print(labels)

    retina_model = build_model()

    retina_extract_output = extract_faces(img_path = train_img_df["path"][0], model = retina_model)
    print(retina_extract_output)

    retina_detect_faces = detect_faces(img_path = train_img_df["path"][0], model = retina_model)
    print(retina_detect_faces)


if __name__ == '__main__':
    main()
