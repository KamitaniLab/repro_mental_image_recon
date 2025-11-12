
from PIL import Image, ImageDraw, ImageFont
import math
import numpy as np


class GroupImageDrawer:
    def __init__(self, conditions, background_color=(255, 255, 255), image_size=(160, 160), 
                 image_margin=(1, 1, 0, 0), group_margin=(20, 0, 20, 0), max_column_size=13, 
                 title_fontcolor="black", title_fontsize=20, title_top_padding=70, title_left_padding=15, 
                 maintain_aspect_ratio=False, image_padding_color=(0, 0, 0), id_show=False, id_fontcolor="black", 
                 id_fontsize=18, image_id_list=[]):
        """
        GroupImageDrawer クラスのコンストラクタ。

        Parameters:
        - conditions (list of dict): A list of dictionaries representing each condition. 
            Each dictionary must contain the following keys:
            - 'title' (str): The title name of the condition.
            - 'images' (list): A list of image file paths, numpy ndarrays, or PIL.Image objects.
        - background_color (tuple): RGB values of the background color.
        - image_size (tuple): The size of each image.
        - image_margin (tuple): Margins around each image (Top, Right, Bottom, Left).
        - group_margin (tuple): Margins around each group (Top, Right, Bottom, Left).
        - max_column_size (int): The maximum number of images per row.
        - title_fontcolor (str or tuple): Font color for titles.
        - title_fontsize (int): Font size for titles.
        - title_top_padding (int): Top padding for titles.
        - title_left_padding (int): Left padding for titles.
        - maintain_aspect_ratio (bool): Whether to preserve the aspect ratio of images.
        - image_padding_color (tuple): Padding color when maintaining aspect ratio.
        - id_show (bool): Whether to display image IDs.
        - id_fontcolor (str or tuple): Font color for image IDs.
        - id_fontsize (int): Font size for image IDs.
        - image_id_list (list): A list of image IDs.
        """
        self.conditions = conditions
        self.background_color = background_color
        self.image_size = image_size
        self.image_margin = image_margin
        self.group_margin = group_margin
        self.max_column_size = max_column_size
        self.title_fontcolor = title_fontcolor if isinstance(title_fontcolor, list) else [title_fontcolor] * len(conditions)
        self.title_fontsize = title_fontsize
        self.title_top_padding = title_top_padding
        self.title_left_padding = title_left_padding
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.image_padding_color = image_padding_color
        self.id_show = id_show
        self.id_fontcolor = id_fontcolor
        self.id_fontsize = id_fontsize
        self.image_id_list = image_id_list

    def draw(self):
        total_image_size = len(self.conditions[0]['images'])
        column_size = min(self.max_column_size, total_image_size)

        # キャンバスの作成
        image = self._create_canvas(total_image_size, column_size)

        # 画像の描画
        self._draw_images(image, total_image_size, column_size)

        # 描画の準備
        image_obj = Image.fromarray(image)
        draw = ImageDraw.Draw(image_obj)

        # タイトルの描画
        self._draw_titles(draw, total_image_size, column_size)

        # 画像IDの描画
        if self.id_show:
            self._draw_image_ids(draw, total_image_size, column_size)

        return image_obj

    def _create_canvas(self, total_image_size, column_size):
        turn_num = math.ceil(total_image_size / float(column_size))
        nImg_row = len(self.conditions) * turn_num
        nImg_col = 1 + column_size  # タイトル列が1つ追加されるため
        size_x = (self.image_size[0] + self.image_margin[0] + self.image_margin[2]) * nImg_row + (self.group_margin[0] + self.group_margin[2]) * turn_num
        size_y = (self.image_size[1] + self.image_margin[1] + self.image_margin[3]) * nImg_col + (self.group_margin[1] + self.group_margin[3])
        return np.ones([size_x, size_y, 3], dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)

    def _draw_images(self, image, total_image_size, column_size):
        for cind, condition in enumerate(self.conditions):
            images = condition['images']
            processed_images = self._process_images(images)

            for tind in range(total_image_size):
                image_obj = processed_images[tind]
                if self.maintain_aspect_ratio:
                    image_obj = self._expand_to_square(image_obj)
                image_obj = image_obj.resize(self.image_size, Image.LANCZOS)

                # 画像位置の計算
                row_index, column_index, turn_index = self._calculate_image_position(cind, tind, column_size)
                x = self.image_margin[0] + self.group_margin[0] + row_index * (self.image_size[0] + self.image_margin[0] + self.image_margin[2])
                x += turn_index * (self.group_margin[0] + self.group_margin[2])
                y = self.image_margin[3] + self.group_margin[3] + column_index * (self.image_size[1] + self.image_margin[1] + self.image_margin[3])
                image[x:(x + self.image_size[0]), y:(y + self.image_size[1]), :] = np.array(image_obj)

    def _draw_titles(self, draw, total_image_size, column_size):
        turn_num = math.ceil(total_image_size / float(column_size))
        for cind, condition in enumerate(self.conditions):
            title = condition['title']
            for turn_index in range(turn_num):
                row_index = cind + turn_index * len(self.conditions)
                x = self.image_margin[0] + self.group_margin[0] + row_index * (self.image_size[0] + self.image_margin[0] + self.image_margin[2])
                x += turn_index * (self.group_margin[0] + self.group_margin[2])
                x += self.title_top_padding
                y = self.title_left_padding
                font = ImageFont.load_default() if self.title_fontsize is None else ImageFont.truetype("DejaVuSans-Bold.ttf", self.title_fontsize)
                draw.text((y, x), title, fill=self.title_fontcolor[cind], font=font)

    def _draw_image_ids(self, draw, total_image_size, column_size):
        for tind in range(total_image_size):
            row_index, column_index, turn_index = self._calculate_image_position(0, tind, column_size, include_condition_offset=False)
            x = self.image_margin[0] + self.group_margin[0] + row_index * (self.image_size[0] + self.image_margin[0] + self.image_margin[2])
            x += turn_index * (self.group_margin[0] + self.group_margin[2])
            x -= self.id_fontsize
            y = self.image_margin[3] + self.group_margin[3] + column_index * (self.image_size[1] + self.image_margin[1] + self.image_margin[3])
            font = ImageFont.load_default() if self.title_fontsize is None else ImageFont.truetype("DejaVuSans-Bold.ttf", self.title_fontsize)
            draw.text((y, x), self.image_id_list[tind], fill=self.id_fontcolor, font=font)

    def _calculate_image_position(self, cind, tind, column_size, include_condition_offset=True):
        row_index = cind + (tind // column_size) * len(self.conditions) if include_condition_offset else (tind // column_size) * len(self.conditions)
        column_index = 1 + tind % column_size
        turn_index = tind // column_size
        return row_index, column_index, turn_index

    def _expand_to_square(self, image_obj):
        width, height = image_obj.size
        size = max(width, height)
        new_image = Image.new("RGB", (size, size), self.image_padding_color)
        new_image.paste(image_obj, ((size - width) // 2, (size - height) // 2))
        return new_image

    def _process_images(self, images):
        processed_images = []
        for img in images:
            if isinstance(img, str):
                image_obj = Image.open(img)
            elif isinstance(img, np.ndarray):
                image_obj = Image.fromarray(img)
            elif isinstance(img, Image.Image):
                image_obj = img
            else:
                raise ValueError("Image must be a filepath, ndarray, or PIL.Image instance.")
            processed_images.append(image_obj.convert("RGB"))
        return processed_images

