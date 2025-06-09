# Importa as bibliotecas necessárias
import cv2
import os       # Para manipulação de caminhos e arquivos do sistema operacional
import requests # Para fazer requisições HTTP (download do arquivo cascade)
import dlib     # Para detecção de marcos faciais
from scipy.spatial import distance as dist # Para calcular a distância euclidiana
from playsound import playsound # Para tocar o som de alerta
import time # Para adicionar um pequeno delay ao som e evitar múltiplos alertas muito rápidos

# --- Constantes para Detecção de Sonolência ---

# --- Relacionadas aos Olhos (EAR) ---
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

# Contadores de frames
EYE_COUNTER = 0
# MOUTH_COUNTER foi removido

# Flag para indicar se o alarme está tocando
ALARM_ON = False
SOUND_ALARM_PATH = "alarm.wav"
last_alarm_time = 0
ALARM_COOLDOWN = 2

# --- Constantes Visuais ---
COR_ROSTO = (0, 200, 0)      # Verde um pouco mais escuro para o retângulo do rosto
COR_OLHOS = (255, 255, 0)   # Ciano para os marcos dos olhos
# COR_BOCA foi removida pois não estamos mais desenhando os marcos da boca explicitamente
COR_TEXTO_INFO = (200, 200, 250) # Branco-azulado claro para EAR
COR_TEXTO_ALERTA = (0, 0, 255)   # Vermelho para o alerta de sonolência
COR_FUNDO_ALERTA = (150, 150, 150) # Cinza claro para o fundo do alerta
ESPESSURA_LINHA_ROSTO = 1
RAIO_MARCO = 1 # Usado para os olhos
FONTE_TEXTO = cv2.FONT_HERSHEY_SIMPLEX
ESCALA_TEXTO_INFO = 0.6
ESCALA_TEXTO_ALERTA = 0.7
ESPESSURA_TEXTO = 2


# --- Potenciais Melhorias Futuras (Comentários) ---
# - CALIBRACAO_AUTOMATICA: Implementar uma rotina de calibração inicial para EYE_AR_THRESH.
# - DETECCAO_INCLINACAO_CABECA: Adicionar deteção de inclinação da cabeça.
# - INTERFACE_GRAFICA: Usar Tkinter, PyQt ou Kivy para uma GUI mais completa.
# - LOGGING_AVANCADO: Usar a biblioteca `logging` para registar eventos.
# - CONFIGURACAO_EXTERNA: Ler constantes (THRESH, CONSEC_FRAMES) de um ficheiro JSON/YAML.

def download_file(url, local_filename):
    """Função genérica para baixar um arquivo."""
    print(f"Ficheiro '{os.path.basename(local_filename)}' não encontrado. A tentar transferir de {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Ficheiro '{os.path.basename(local_filename)}' transferido e guardado em: {local_filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erro ao transferir o ficheiro '{os.path.basename(local_filename)}': {e}")
        if os.path.exists(local_filename): os.remove(local_filename)
        return False
    except IOError as e:
        print(f"Erro ao guardar o ficheiro '{os.path.basename(local_filename)}': {e}")
        return False

def setup_projeto(base_dir):
    """
    Verifica e cria a estrutura de pastas e baixa os arquivos necessários.
    Retorna os caminhos para os arquivos ou None em caso de erro.
    """
    # Para Haar Cascade (detecção de rosto)
    haarcascades_dir_name = "haarcascades"
    cascade_file_name = "haarcascade_frontalface_default.xml"
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    haarcascades_path = os.path.join(base_dir, haarcascades_dir_name)
    cascade_file_path = os.path.join(haarcascades_path, cascade_file_name)

    if not os.path.exists(haarcascades_path):
        try:
            os.makedirs(haarcascades_path)
            print(f"Pasta '{haarcascades_dir_name}' criada em: {haarcascades_path}")
        except OSError as e:
            print(f"Erro ao criar a pasta '{haarcascades_dir_name}': {e}")
            return None, None
    
    if not os.path.exists(cascade_file_path):
        if not download_file(cascade_url, cascade_file_path):
            return None, None
    else:
        print(f"Ficheiro '{cascade_file_name}' já existe em: {cascade_file_path}")

    # Para Dlib Shape Predictor (marcos faciais)
    dlib_models_dir_name = "dlib_models"
    shape_predictor_file_name = "shape_predictor_68_face_landmarks.dat"
    shape_predictor_dat_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"

    dlib_models_path = os.path.join(base_dir, dlib_models_dir_name)
    shape_predictor_file_path = os.path.join(dlib_models_path, shape_predictor_file_name)

    if not os.path.exists(dlib_models_path):
        try:
            os.makedirs(dlib_models_path)
            print(f"Pasta '{dlib_models_dir_name}' criada em: {dlib_models_path}")
        except OSError as e:
            print(f"Erro ao criar a pasta '{dlib_models_dir_name}': {e}")
            return cascade_file_path, None 

    if not os.path.exists(shape_predictor_file_path):
        print(f"--------------------------------------------------------------------")
        print(f"ATENÇÃO: O ficheiro '{shape_predictor_file_name}' não foi encontrado em '{dlib_models_path}'.")
        print(f"Por favor, transfira-o de uma fonte confiável (ex: site do dlib ou repositório de modelos do dlib)")
        print(f"e coloque-o na pasta: {dlib_models_path}")
        print(f"Um link comum para o ficheiro compactado (.bz2) é: {shape_predictor_dat_url}")
        print(f"Precisará de o descompactar para obter o ficheiro .dat.")
        print(f"--------------------------------------------------------------------")
        return cascade_file_path, None
    else:
         print(f"Ficheiro '{shape_predictor_file_name}' já existe em: {shape_predictor_file_path}")

    return cascade_file_path, shape_predictor_file_path


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# def mouth_aspect_ratio(mouth): foi removida

def play_alarm_sound():
    global last_alarm_time 
    current_time = time.time()
    if (current_time - last_alarm_time > ALARM_COOLDOWN):
        last_alarm_time = current_time 
        print("ALERTA DE SONOLÊNCIA!") 
        try:
            if SOUND_ALARM_PATH and os.path.exists(SOUND_ALARM_PATH):
                playsound(SOUND_ALARM_PATH, block=False) 
            else:
                if os.name == 'nt': 
                    import winsound
                    winsound.Beep(1000, 500) 
                else:
                    print("\a") 
        except Exception as e:
            print(f"Erro ao tocar o som de alerta: {e}")
            if os.name == 'nt':
                import winsound
                winsound.Beep(1000, 500)
            else:
                print("\a")


def iniciar_detector_sonolencia():
    global EYE_COUNTER, ALARM_ON, last_alarm_time # MOUTH_COUNTER removido dos globais

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path, shape_predictor_path = setup_projeto(base_dir)

    if not cascade_path or not shape_predictor_path:
        print("Falha na configuração dos ficheiros de modelo. A encerrar.")
        return

    try:
        detector_rosto_dlib = dlib.get_frontal_face_detector()
        original_cwd = os.getcwd()
        predictor_dir = os.path.dirname(shape_predictor_path)
        predictor_filename = os.path.basename(shape_predictor_path)
        if not os.path.isdir(predictor_dir): raise IOError(f"O diretório '{predictor_dir}' não foi encontrado.")
        try:
            os.chdir(predictor_dir)
            preditor_marcos = dlib.shape_predictor(predictor_filename)
        finally:
            os.chdir(original_cwd)
        if preditor_marcos is None : raise RuntimeError("Falha ao carregar preditor de marcos.")

    except Exception as e: 
        print(f"Erro ao carregar o detetor de rosto do dlib ou o preditor de marcos: {e}")
        return

    (lStart, lEnd) = (42, 48) 
    (rStart, rEnd) = (36, 42) 
    # (mStart, mEnd) foi removido

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmara.")
        return

    print("\nCâmara iniciada. Deteção de sonolência ativa.")
    print(f"Limiar EAR: {EYE_AR_THRESH}, Frames Olhos: {EYE_AR_CONSEC_FRAMES}")
    # Print do Limiar MAR foi removido
    print("Pressione 'q' na janela do vídeo para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler frame. A encerrar.")
            break

        frame = cv2.flip(frame, 1) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector_rosto_dlib(gray, 0) 

        for rect in rects: 
            (x_rect, y_rect, w_rect, h_rect) = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), COR_ROSTO, ESPESSURA_LINHA_ROSTO)

            shape = preditor_marcos(gray, rect)
            coords = []
            for i in range(0, 68): 
                coords.append((shape.part(i).x, shape.part(i).y))
            
            olho_esquerdo_coords = coords[lStart:lEnd]
            olho_direito_coords = coords[rStart:rEnd]
            # boca_coords foi removida

            ear_esquerdo = eye_aspect_ratio(olho_esquerdo_coords)
            ear_direito = eye_aspect_ratio(olho_direito_coords)
            ear = (ear_esquerdo + ear_direito) / 2.0 

            # mar = mouth_aspect_ratio(boca_coords) foi removido

            for (ex, ey) in olho_esquerdo_coords: cv2.circle(frame, (ex, ey), RAIO_MARCO, COR_OLHOS, -1)
            for (ex, ey) in olho_direito_coords: cv2.circle(frame, (ex, ey), RAIO_MARCO, COR_OLHOS, -1)
            # Desenho dos marcos da boca foi removido
            
            info_y_start = y_rect - 10 if y_rect - 10 > 10 else y_rect + h_rect + 20
            cv2.putText(frame, f"EAR: {ear:.2f}", (x_rect, info_y_start),
                        FONTE_TEXTO, ESCALA_TEXTO_INFO, COR_TEXTO_INFO, 1)
            # Desenho do MAR foi removido

            if ear < EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON: 
                        print("Sonolência detetada (olhos)!")
                    ALARM_ON = True 
            else:
                EYE_COUNTER = 0
                # Se apenas os olhos controlam o alarme, podemos resetar ALARM_ON aqui
                if ALARM_ON: # Se o alarme estava ativo, podemos opcionalmente indicar que parou
                    print("Condição de sonolência não mais detetada (olhos).")
                ALARM_ON = False

            # Lógica de deteção de bocejo foi removida

            if ALARM_ON: # Agora ALARM_ON é controlado apenas pelos olhos
                play_alarm_sound() 
                texto_alerta = "SONOLENCIA DETECTADA!"
                (largura_texto_alerta, altura_texto_alerta), _ = cv2.getTextSize(texto_alerta, FONTE_TEXTO, ESCALA_TEXTO_ALERTA, ESPESSURA_TEXTO)
                pos_alerta_x = x_rect + (w_rect - largura_texto_alerta) // 2
                pos_alerta_y = y_rect + h_rect + altura_texto_alerta + 10
                if pos_alerta_y + altura_texto_alerta > frame.shape[0]: 
                    pos_alerta_x = (frame.shape[1] - largura_texto_alerta) // 2 
                    pos_alerta_y = frame.shape[0] - 20

                cv2.rectangle(frame, (pos_alerta_x - 5, pos_alerta_y - altura_texto_alerta - 5), 
                              (pos_alerta_x + largura_texto_alerta + 5, pos_alerta_y + 5), COR_FUNDO_ALERTA, -1)
                cv2.putText(frame, texto_alerta, (pos_alerta_x, pos_alerta_y),
                            FONTE_TEXTO, ESCALA_TEXTO_ALERTA, COR_TEXTO_ALERTA, ESPESSURA_TEXTO)
            
            # A condição para resetar ALARM_ON foi simplificada, pois agora só depende de EYE_COUNTER
            # if EYE_COUNTER == 0 and ALARM_ON: # Já tratado acima quando EYE_COUNTER é resetado
            #     print("Condição de sonolência não mais detetada.")
            #     ALARM_ON = False

        cv2.imshow("Detector de Sonolencia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recursos libertados.")

if __name__ == "__main__":
    try:
        import requests
        import dlib
        from scipy.spatial import distance
        from playsound import playsound
    except ImportError as e:
        print("--------------------------------------------------------------------")
        print(f"ERRO DE IMPORTAÇÃO: {e}")
        print("Uma ou mais bibliotecas necessárias não estão instaladas.")
        print("Por favor, instale-as executando:")
        print("pip install opencv-python requests dlib scipy playsound")
        print("Se a instalação do 'dlib' falhar, procure por instruções específicas para seu sistema.")
        print("--------------------------------------------------------------------")
        exit()

    iniciar_detector_sonolencia()
