# script para renombrar imagenes con la misma nomenclatura
import os

# Rutas de las carpetas
ruta_base = "C:/Users/jbbag/OneDrive/Escritorio/Frutas-seleccionadas"
carpetas = ["Test", "Training"]

# Extensiones válidas para imágenes
extensiones_validas = (".jpg", ".png", ".jpeg")

# Procesar cada carpeta (Test y Training)
for carpeta_principal in carpetas:
    ruta_actual = os.path.join(ruta_base, carpeta_principal)

    if not os.path.exists(ruta_actual):
        print(f" La carpeta '{carpeta_principal}' no existe. Verifica la ruta y vuelve a intentarlo.")
        continue

    # Recorrer cada subcarpeta de frutas dentro de Test y Training
    for subcarpeta in os.listdir(ruta_actual):
        ruta_fruta = os.path.join(ruta_actual, subcarpeta)

        if os.path.isdir(ruta_fruta):
            print(f"\n Procesando carpeta: {carpeta_principal}/{subcarpeta}")

            # Obtener archivos de imagen dentro de la subcarpeta
            archivos = [f for f in os.listdir(ruta_fruta) if f.lower().endswith(extensiones_validas)]
            archivos.sort()

            if not archivos:
                print(f" No hay imágenes en {carpeta_principal}/{subcarpeta}, saltando...")
                continue

            # **Registrar nombres existentes antes de renombrar**
            nombres_existentes = set(archivos)
            contador = 1

            # **Buscar el número más alto existente para continuar**
            for archivo in archivos:
                nombre_sin_ext, extension = os.path.splitext(archivo)
                partes = nombre_sin_ext.split('_')
                if len(partes) == 2 and partes[1].isdigit():
                    contador = max(contador, int(partes[1]) + 1)

            # Renombrar imágenes evitando conflictos
            for archivo in archivos:
                extension = os.path.splitext(archivo)[1]
                nuevo_nombre = f"{subcarpeta}_{contador:03}{extension}"

                ruta_original = os.path.join(ruta_fruta, archivo)
                ruta_nueva = os.path.join(ruta_fruta, nuevo_nombre)

                # **Evitar conflictos con nombres existentes**
                while nuevo_nombre in nombres_existentes:
                    contador += 1
                    nuevo_nombre = f"{subcarpeta}_{contador:03}{extension}"
                    ruta_nueva = os.path.join(ruta_fruta, nuevo_nombre)

                # **Renombrar archivo**
                if os.path.exists(ruta_original):
                    os.rename(ruta_original, ruta_nueva)
                    print(f" Renombrado: {archivo} → {nuevo_nombre}")
                    nombres_existentes.add(nuevo_nombre)
                    contador += 1
                else:
                    print(f"⚠️ No se encontró {ruta_original}, saltando...")

print("\n Renombramiento completado sin conflictos.")