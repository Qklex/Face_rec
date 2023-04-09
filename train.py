import face_recognition
import os
import pickle
def train(name):
    # Получаем список файлов из папки с фотографиями
    path = "data/"+ name
    image_list = os.listdir(path)

    # Ограничиваем количество сохраняемых фотографий до 50
    if len(image_list) > 50:
        image_list = image_list[:50]

    # Загружаем изображения и создаем список их дескрипторов
    known_face_descriptors = []
    for image_name in image_list:
        image_path = os.path.join(path, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_descriptors.append(face_encoding)

    # Сохраняем список дескрипторов в файл

    data = {
        "known_face_descriptors": known_face_descriptors,
        "name": name
    }


    with open('user_model.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



