import numpy as np
from core.conversion_espace_couleur import ConversionEspaceCouleur

class DescripteurForme:
    
    @staticmethod
    def descripteur_forme(image, descriptor, filter):
        """
        Compute the shape descriptor of an image.
        
        Args:
            image (np.array): Input image.
            descriptor (str): Type of shape descriptor.
            filter (str): Type of filter to use ('sobel', 'prewitt', 'scharr').
            
        Returns:
            np.array: Shape descriptor.
        """
        if descriptor == "HOG":
            return DescripteurForme.hog(image, filter_type=filter)
        elif descriptor == "HOPN":
            return DescripteurForme.hopn(image, filter_type=filter)
        elif descriptor == "HBO":
            return DescripteurForme.hbo(image, filter_type=filter)
        elif descriptor == "HBOQ":
            return DescripteurForme.hboq(image, filter_type=filter)
        return np.array([])
    

    @staticmethod
    def get_filter(filter):
        """
        Get the convolution filters for the specified type.
        
        Args:
            type (str): Type of filter to use ('sobel', 'prewitt', 'scharr').
            
        Returns:
            tuple: Convolution filters for the specified type.
        """
        if filter is None:
            return None, None
        filter = filter.lower()
        if filter == 'sobel':
            gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif filter == 'prewitt':
            gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            gy_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif filter == 'scharr':
            gx_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
            gy_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        else:
            raise ValueError(f"Invalid filter_type {filter}. Choose 'sobel', 'prewitt', or 'scharr'.")
        
        return gx_kernel, gy_kernel
        

    @staticmethod
    def apply_filter(image, kernel):
        """
        Apply a convolution filter to the image using the given kernel.

        Parameters:
            image (ndarray): Input image (grayscale).
            kernel (ndarray): Convolution kernel.

        Returns:
            ndarray: Filtered image.
        """
        return np.convolve(image.ravel(), kernel.ravel(), mode='same').reshape(image.shape)

    @staticmethod
    def hog(image, cell_size=(8, 8), block_size=(2, 2), bins=9, filter_type='sobel'):
        """
        Compute the Histogram of Oriented Gradients (HOG) descriptor for an input image.

        Parameters:
            image (ndarray): Input image (grayscale or RGB).
            cell_size (tuple): Size of each cell (in pixels).
            block_size (tuple): Size of each block (in cells).
            bins (int): Number of orientation bins.
            filter_type (str): Type of filter to use ('sobel', 'prewitt', 'scharr').

        Returns:
            ndarray: HOG feature vector.
        """
        if image.ndim == 3:
            image = ConversionEspaceCouleur.rgb_to_gray_basic(image)

        gx_kernel, gy_kernel = DescripteurForme.get_filter(filter_type)

        gx = DescripteurForme.apply_filter(image, gx_kernel)
        gy = DescripteurForme.apply_filter(image, gy_kernel)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180

        n_cells_x = image.shape[1] // cell_size[1]
        n_cells_y = image.shape[0] // cell_size[0]
        histograms = np.zeros((n_cells_y, n_cells_x, bins))

        bin_edges = np.linspace(0, 180, bins + 1)

        # Vectorized cell histogram computation
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_magnitude = magnitude[i * cell_size[0]:(i + 1) * cell_size[0],
                                        j * cell_size[1]:(j + 1) * cell_size[1]]
                cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                                            j * cell_size[1]:(j + 1) * cell_size[1]]

                hist, _ = np.histogram(cell_orientation, bins=bin_edges, weights=cell_magnitude)
                histograms[i, j, :] = hist

        n_blocks_x = n_cells_x - block_size[1] + 1
        n_blocks_y = n_cells_y - block_size[0] + 1
        normalized_blocks = []

        # Vectorized block normalization
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = histograms[i:i + block_size[0], j:j + block_size[1], :]
                block_vector = block.ravel()
                norm = np.linalg.norm(block_vector, ord=2) + 1e-6
                normalized_blocks.append(block_vector / norm)

        hog_descriptor = np.hstack(normalized_blocks)

        return hog_descriptor

    @staticmethod
    def hopn(image, cell_size=(8, 8), bins=9, filter_type='sobel'):
        """
        Compute the Histogram of Oriented Point Normals (HOPN) descriptor.

        Parameters:
            image (ndarray): Input image (grayscale or RGB).
            cell_size (tuple): Size of each cell (in pixels).
            bins (int): Number of orientation bins.
            filter_type (str): Type of filter to use ('sobel', 'prewitt', 'scharr').

        Returns:
            ndarray: HOPN feature vector.
        """
        if image.ndim == 3:
            image = ConversionEspaceCouleur.rgb_to_gray_basic(image)

        gx_kernel, gy_kernel = DescripteurForme.get_filter(filter_type)

        gx = DescripteurForme.apply_filter(image, gx_kernel)
        gy = DescripteurForme.apply_filter(image, gy_kernel)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180

        n_cells_x = image.shape[1] // cell_size[1]
        n_cells_y = image.shape[0] // cell_size[0]
        histograms = np.zeros((n_cells_y, n_cells_x, bins))

        bin_edges = np.linspace(0, 180, bins + 1)

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_magnitude = magnitude[i * cell_size[0]:(i + 1) * cell_size[0],
                                        j * cell_size[1]:(j + 1) * cell_size[1]]
                cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                                            j * cell_size[1]:(j + 1) * cell_size[1]]

                hist, _ = np.histogram(cell_orientation, bins=bin_edges, weights=cell_magnitude)
                histograms[i, j, :] = hist

        return histograms.ravel()

    @staticmethod
    def hbo(image, cell_size=(8, 8), filter_type='sobel'):
        """
        Compute the Histogram of Binary Orientations (HBO) descriptor.

        Parameters:
            image (ndarray): Input image (grayscale or RGB).
            cell_size (tuple): Size of each cell (in pixels).
            filter_type (str): Type of filter to use ('sobel', 'prewitt', 'scharr').

        Returns:
            ndarray: HBO feature vector.
        """
        if image.ndim == 3:
            image = ConversionEspaceCouleur.rgb_to_gray_basic(image)

        gx_kernel, gy_kernel = DescripteurForme.get_filter(filter_type)

        gx = DescripteurForme.apply_filter(image, gx_kernel)
        gy = DescripteurForme.apply_filter(image, gy_kernel)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180

        n_cells_x = image.shape[1] // cell_size[1]
        n_cells_y = image.shape[0] // cell_size[0]
        histograms = np.zeros((n_cells_y, n_cells_x))

        # Binary orientation computation
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                                            j * cell_size[1]:(j + 1) * cell_size[1]]

                histograms[i, j] = (cell_orientation < 90).sum() / cell_orientation.size

        return histograms.ravel()

    @staticmethod
    def hboq(image, cell_size=(8, 8), filter_type='sobel'):
        """
        Compute the Quantized Histogram of Binary Orientations (HBOQ) descriptor.

        Parameters:
            image (ndarray): Input image (grayscale or RGB).
            cell_size (tuple): Size of each cell (in pixels).
            filter_type (str): Type of filter to use ('sobel', 'prewitt', 'scharr').

        Returns:
            ndarray: HBOQ feature vector.
        """
        if image.ndim == 3:
            image = ConversionEspaceCouleur.rgb_to_gray_basic(image)

        gx_kernel, gy_kernel = DescripteurForme.get_filter(filter_type)

        gx = DescripteurForme.apply_filter(image, gx_kernel)
        gy = DescripteurForme.apply_filter(image, gy_kernel)

        orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180

        n_cells_x = image.shape[1] // cell_size[1]
        n_cells_y = image.shape[0] // cell_size[0]
        histograms = np.zeros((n_cells_y, n_cells_x, 2))

        # Quantized binary orientation computation
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                                            j * cell_size[1]:(j + 1) * cell_size[1]]

                histograms[i, j, 0] = (cell_orientation < 90).sum() / cell_orientation.size
                histograms[i, j, 1] = (cell_orientation >= 90).sum() / cell_orientation.size

        return histograms.ravel()