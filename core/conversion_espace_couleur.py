import numpy as np

class ConversionEspaceCouleur:
    
    @staticmethod
    def rgb(image):
        """
        Convertit une image RGB en une image RGB.
        
        Args:
            image (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        
        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        return image
    
    @staticmethod
    def rgb_normalized(image):
        """
        Convertit une image RGB en une image RGB normalisée.
        
        Args:
            image (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        
        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB normalisée.
        """
        return image / 255.0

    @staticmethod
    def rgb_to_gray_basic(rgb):
        """
        Convertit une image RGB en une image en niveaux de gris en moyennant les valeurs des canaux R, G, B.
        
        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        
        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width) représentant l'image en niveaux de gris.
        """
        return np.mean(rgb, axis=2)
    
    @staticmethod
    def rgb_to_gray_709(rgb):
        """
        Convertit une image RGB en une image en niveaux de gris en utilisant les coefficients de la norme Rec. 709.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width) représentant l'image en niveaux de gris.
        """
        return np.dot(rgb, [0.2126, 0.7152, 0.0722])
    
    @staticmethod
    def rgb_to_gray_601(rgb):
        """
        Convertit une image RGB en une image en niveaux de gris en utilisant les coefficients de la norme Rec. 601.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width) représentant l'image en niveaux de gris.
        """
        return np.dot(rgb, [0.299, 0.587, 0.114])
    
    @staticmethod
    def rgb_to_yiq(rgb):
        """
        Convertit une image RGB en une image YIQ.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image YIQ.
        """
        yiq = np.zeros_like(rgb, dtype=np.float32)
        yiq[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        yiq[:, :, 1] = 0.595716 * rgb[:, :, 0] - 0.274453 * rgb[:, :, 1] - 0.321263 * rgb[:, :, 2]
        yiq[:, :, 2] = 0.211456 * rgb[:, :, 0] - 0.522591 * rgb[:, :, 1] + 0.311135 * rgb[:, :, 2]
        return yiq
    
    @staticmethod
    def rgb_to_yuv(rgb):
        """
        Convertit une image RGB en une image YUV.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image YUV.
        """
        yuv = np.zeros_like(rgb, dtype=np.float32)
        yuv[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        yuv[:, :, 1] = -0.147 * rgb[:, :, 0] - 0.289 * rgb[:, :, 1] + 0.436 * rgb[:, :, 2]
        yuv[:, :, 2] = 0.615 * rgb[:, :, 0] - 0.515 * rgb[:, :, 1] - 0.100 * rgb[:, :, 2]
        return yuv
    
    @staticmethod
    def rgb_to_l1l2l3(rgb):
        """
        Convertit une image RGB en une image L1L2L3.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image L1L2L3.
        """
        l1l2l3 = np.zeros_like(rgb, dtype=np.float32)
        l1l2l3[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        l1l2l3[:, :, 1] = 0.5 * rgb[:, :, 0] + 0.5 * rgb[:, :, 1]
        l1l2l3[:, :, 2] = 0.866 * rgb[:, :, 1] - 0.866 * rgb[:, :, 2]
        return l1l2l3
    
    @staticmethod
    def rgb_to_normalized_rgb(rgb):
        """
        Convertit une image RGB en une image RGB normalisée.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB normalisée.
        """
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        total = R + G + B + 1e-6
        r_norm = R / total
        g_norm = G / total
        b_norm = B / total
        return np.stack((r_norm, g_norm, b_norm), axis=2)
    
    @staticmethod
    def rgb_to_hsv(rgb):
        """
        Convertit une image RGB en une image HSV.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image HSV.
        """
        rgb = rgb / 255.0
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        delta = max_val - min_val

        hue = np.zeros_like(max_val)
        mask = delta != 0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        hue[mask & (max_val == r)] = (60 * (g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)]) % 360
        hue[mask & (max_val == g)] = (60 * (b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)]) + 120
        hue[mask & (max_val == b)] = (60 * (r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)]) + 240

        saturation = np.zeros_like(max_val)
        saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

        value = max_val

        hsv = np.stack([hue, saturation, value], axis=-1)
        return hsv
    
    @staticmethod
    def rgb_to_hsl(rgb):
        """
        Convertit une image RGB en une image HSL.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image HSL.
        """
        rgb = rgb / 255.0
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        delta = max_val - min_val

        # Luminosité (Lightness)
        lightness = (max_val + min_val) / 2

        # Saturation
        saturation = np.zeros_like(max_val)
        mask = delta != 0
        saturation[mask & (lightness < 0.5)] = delta[mask & (lightness < 0.5)] / (max_val[mask & (lightness < 0.5)] + min_val[mask & (lightness < 0.5)])
        saturation[mask & (lightness >= 0.5)] = delta[mask & (lightness >= 0.5)] / (2.0 - max_val[mask & (lightness >= 0.5)] - min_val[mask & (lightness >= 0.5)])

        # Teinte (Hue)
        hue = np.zeros_like(max_val)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hue[mask & (max_val == r)] = (60 * (g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)]) % 360
        hue[mask & (max_val == g)] = (60 * (b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)]) + 120
        hue[mask & (max_val == b)] = (60 * (r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)]) + 240

        hue[hue < 0] += 360  # Corriger les valeurs de teinte négatives

        hsl = np.stack([hue, saturation, lightness], axis=-1)
        return hsl
    
    @staticmethod
    def rgb_to_lab(rgb):
        """
        Convertit une image RGB en une image Lab.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image Lab.
        """
        def f(t):
            delta = 6 / 29
            return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4 / 29))

        def xyz_to_lab(xyz):
            ref_X, ref_Y, ref_Z = 0.95047, 1.00000, 1.08883  # D65 reference white
            X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

            L = 116 * f(Y / ref_Y) - 16
            a = 500 * (f(X / ref_X) - f(Y / ref_Y))
            b = 200 * (f(Y / ref_Y) - f(Z / ref_Z))

            return np.stack([L, a, b], axis=-1)

        rgb = rgb / 255.0
        # Matrice de conversion RGB vers XYZ
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

        xyz = np.dot(rgb, M.T)
        return xyz_to_lab(xyz)
    
    @staticmethod
    def rgb_to_luv(rgb):
        """
        Convertit une image RGB en une image Luv.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image Luv.
        """
        def xyz_to_luv(xyz):
            ref_X, ref_Y, ref_Z = 0.95047, 1.00000, 1.08883  # D65 référence
            X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

            denom = X + 15 * Y + 3 * Z
            u_prime = np.where(denom != 0, (4 * X) / denom, 0)
            v_prime = np.where(denom != 0, (9 * Y) / denom, 0)

            Y_ratio = Y / ref_Y
            L = np.where(Y_ratio > (6/29)**3, 116 * np.cbrt(Y_ratio) - 16, (29/3)**3 * Y_ratio)

            u = 13 * L * (u_prime - 0.19783000664283)
            v = 13 * L * (v_prime - 0.46831999493879)

            return np.stack([L, u, v], axis=-1)

        rgb = rgb / 255.0
        # Matrice de conversion RGB vers XYZ
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

        xyz = np.dot(rgb, M.T)
        return xyz_to_luv(xyz)
    
    @staticmethod
    def rgb_to_cmyk(rgb):
        """
        Convertit une image RGB en une image CMYK.

        Args:
            rgb (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 4) représentant l'image CMYK.
        """
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        K = 1 - np.max(rgb / 255.0, axis=2)
        C = (1 - R / 255.0 - K) / (1 - K)
        M = (1 - G / 255.0 - K) / (1 - K)
        Y = (1 - B / 255.0 - K) / (1 - K)
        return np.stack((C, M, Y, K), axis=2)
    
    @staticmethod
    def yuv_to_rgb(yuv):
        """
        Convertit une image YUV en une image RGB.

        Args:
            yuv (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image YUV.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        rgb = np.zeros_like(yuv, dtype=np.float32)
        rgb[:, :, 0] = yuv[:, :, 0] + 1.13983 * yuv[:, :, 2]
        rgb[:, :, 1] = yuv[:, :, 0] - 0.39465 * yuv[:, :, 1] - 0.58060 * yuv[:, :, 2]
        rgb[:, :, 2] = yuv[:, :, 0] + 2.03211 * yuv[:, :, 1]
        return rgb
    
    @staticmethod
    def l1l2l3_to_rgb(img):
        """
        Convertit une image L1L2L3 en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image L1L2L3.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        rgb = np.zeros_like(img, dtype=np.float32)
        rgb[:, :, 0] = img[:, :, 0] + 0.114 * img[:, :, 2] + 0.701 * img[:, :, 1]
        rgb[:, :, 1] = img[:, :, 0] - 0.114 * img[:, :, 2] - 0.5957 * img[:, :, 1]
        rgb[:, :, 2] = img[:, :, 0] + 0.413 * img[:, :, 2] - 0.312 * img[:, :, 1]
        return rgb
    
    @staticmethod
    def normalized_rgb_to_rgb(img):
        """
        Convertit une image RGB normalisée en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image RGB normalisée.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        total = r + g + b + 1e-6
        R = r / total
        G = g / total
        B = b / total
        return np.stack((R, G, B), axis=2)
    
    @staticmethod
    def hsv_to_rgb(img):
        """
        Convertit une image HSV en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image HSV.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        h, s, v = img[..., 0], img[..., 1], img[..., 2]
        c = s * v
        x = c * (1 - np.abs((h / 60.0) % 2 - 1))
        m = v - c

        rgb = np.zeros_like(img)

        mask = (h < 60)
        rgb[mask] = np.stack([c, x, 0], axis=-1)[mask]
        mask = (h >= 60) & (h < 120)
        rgb[mask] = np.stack([x, c, 0], axis=-1)[mask]
        mask = (h >= 120) & (h < 180)
        rgb[mask] = np.stack([0, c, x], axis=-1)[mask]
        mask = (h >= 180) & (h < 240)
        rgb[mask] = np.stack([0, x, c], axis=-1)[mask]
        mask = (h >= 240) & (h < 300)
        rgb[mask] = np.stack([x, 0, c], axis=-1)[mask]
        mask = (h >= 300)
        rgb[mask] = np.stack([c, 0, x], axis=-1)[mask]

        return (rgb + m[..., np.newaxis]) * 255.0
    
    @staticmethod
    def hsl_to_rgb(img):
        """
        Convertit une image HSL en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image HSL.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        h, s, l = img[..., 0], img[..., 1], img[..., 2]
        
        c = (1 - np.abs(2 * l - 1)) * s  # Chroma
        x = c * (1 - np.abs((h / 60.0) % 2 - 1))
        m = l - c / 2
        
        rgb = np.zeros_like(img)

        mask = (h < 60)
        rgb[mask] = np.stack([c, x, 0], axis=-1)[mask]
        mask = (h >= 60) & (h < 120)
        rgb[mask] = np.stack([x, c, 0], axis=-1)[mask]
        mask = (h >= 120) & (h < 180)
        rgb[mask] = np.stack([0, c, x], axis=-1)[mask]
        mask = (h >= 180) & (h < 240)
        rgb[mask] = np.stack([0, x, c], axis=-1)[mask]
        mask = (h >= 240) & (h < 300)
        rgb[mask] = np.stack([x, 0, c], axis=-1)[mask]
        mask = (h >= 300)
        rgb[mask] = np.stack([c, 0, x], axis=-1)[mask]

        return (rgb + m[..., np.newaxis]) * 255.0
    
    @staticmethod
    def lab_to_rgb(img):
        """
        Convertit une image Lab en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image Lab.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        def f_inv(t):
            delta = 6 / 29
            return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

        def lab_to_xyz(lab):
            L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

            # Reference white D65
            ref_X, ref_Y, ref_Z = 0.95047, 1.00000, 1.08883

            fy = (L + 16) / 116
            fx = fy + (a / 500)
            fz = fy - (b / 200)

            X = ref_X * f_inv(fx)
            Y = ref_Y * f_inv(fy)
            Z = ref_Z * f_inv(fz)

            return np.stack([X, Y, Z], axis=-1)

        def xyz_to_rgb(xyz):
            # Conversion matrix from XYZ to linear RGB (sRGB)
            M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]])

            rgb = np.dot(xyz, M.T)

            # Apply gamma correction (linear RGB to sRGB)
            rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1 / 2.4)) - 0.055)
            return np.clip(rgb, 0, 1)

        xyz = lab_to_xyz(img)
        rgb = xyz_to_rgb(xyz)

        return (rgb * 255).astype(np.uint8)
    
    @staticmethod
    def luv_to_rgb(img):
        """
        Convertit une image Luv en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 3) représentant l'image Luv.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        def luv_to_xyz(luv):
            L, u, v = luv[..., 0], luv[..., 1], luv[..., 2]

            # D65 reference white
            ref_X, ref_Y, ref_Z = 0.95047, 1.00000, 1.08883
            u_prime_ref = (4 * ref_X) / (ref_X + 15 * ref_Y + 3 * ref_Z)
            v_prime_ref = (9 * ref_Y) / (ref_X + 15 * ref_Y + 3 * ref_Z)

            u_prime = u / (13 * L) + u_prime_ref
            v_prime = v / (13 * L) + v_prime_ref

            Y = np.where(L > 8, ((L + 16) / 116) ** 3, L / 903.3)
            X = -(9 * Y * u_prime) / ((u_prime - 4) * v_prime - u_prime * v_prime)
            Z = (9 * Y - 15 * v_prime * Y - v_prime * X) / (3 * v_prime)

            return np.stack([X, Y, Z], axis=-1)

        def xyz_to_rgb(xyz):
            # Conversion matrix from XYZ to linear RGB (sRGB)
            M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]])

            rgb = np.dot(xyz, M.T)

            # Apply gamma correction (linear RGB to sRGB)
            rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1 / 2.4)) - 0.055)
            return np.clip(rgb, 0, 1)

        xyz = luv_to_xyz(img)
        rgb = xyz_to_rgb(xyz)

        return (rgb * 255).astype(np.uint8)
    
    @staticmethod
    def cmyk_to_rgb(img):
        """
        Convertit une image CMYK en une image RGB.

        Args:
            img (np.ndarray): Un tableau NumPy de forme (length, width, 4) représentant l'image CMYK.

        Returns:
            np.ndarray: Un tableau NumPy de forme (length, width, 3) représentant l'image RGB.
        """
        C, M, Y, K = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        R = 255 * (1 - C) * (1 - K)
        G = 255 * (1 - M) * (1 - K)
        B = 255 * (1 - Y) * (1 - K)
        return np.stack((R, G, B), axis=2)
    
    
    def rgb_indexer(image, qr=2, qg=2, qb=2):
        """
        Calcule l'index d'une image RGB.
        
        Args:
            image (np.array): Image à traiter.
            qr (int): Quantification du canal rouge.
            qg (int): Quantification du canal vert.
            qb (int): Quantification du canal bleu.
            
        Returns:
            np.array: Index de l'image indexée.
        """
        pas_r = 256 // qr
        pas_g = 256 // qg
        pas_b = 256 // qb

        index_r = image[:, :, 0] // pas_r
        index_g = image[:, :, 1] // pas_g
        index_b = image[:, :, 2] // pas_b

        # Correction du calcul de l'index
        result_index = (index_r * (qg * qb)) + (index_g * qb) + index_b
        return result_index