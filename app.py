import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import itertools
import functools
from itertools import chain

# 1. Функция визуализации последовательностей многоугольников
def visualize_polygons(poly_iter, N=None):
    """Визуализирует первые N многоугольников из итератора или все, если N не указан."""
    if N is not None:
        poly_iter = itertools.islice(poly_iter, N)
    polygons = list(poly_iter)
    fig, ax = plt.subplots()
    for poly in polygons:
        points = list(poly)
        patch = Polygon(points, closed=True, fill=None, edgecolor='b')
        ax.add_patch(patch)
    if polygons:
        all_points = [p for poly in polygons for p in poly]
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
    plt.show()

# 2. Генераторы бесконечных последовательностей многоугольников
def gen_rectangle():
    """Генерирует прямоугольники с координатами (2n,0), (2n,1), (2n+1,1), (2n+1,0)."""
    n = 0
    while True:
        yield ((2*n, 0), (2*n, 1), (2*n+1, 1), (2*n+1, 0))
        n += 1

def gen_triangle():
    """Генерирует равносторонние треугольники с основанием от (2n,0) до (2n+1,0)."""
    n = 0
    while True:
        yield ((2*n, 0), (2*n+1, 0), (2*n + 0.5, math.sqrt(3)/2))
        n += 1

def gen_hexagon():
    """Генерирует правильные шестиугольники с центром в (3n,0) и длиной стороны 1."""
    n = 0
    while True:
        cx = 3 * n
        s = 1
        yield tuple((cx + s * math.cos(2 * math.pi * k / 6),
                     s * math.sin(2 * math.pi * k / 6)) for k in range(6))
        n += 1

# 3. Трансформации для многоугольников
def tr_translate(dx, dy, poly):
    """Сдвигает многоугольник на вектор (dx, dy)."""
    return tuple((x + dx, y + dy) for (x, y) in poly)

def tr_rotate(theta, poly, center=(0, 0)):
    """Поворачивает многоугольник на угол theta вокруг центра."""
    cx, cy = center
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return tuple((cx + (x - cx) * cos_theta - (y - cy) * sin_theta,
                  cy + (x - cx) * sin_theta + (y - cy) * cos_theta)
                 for (x, y) in poly)

def tr_symmetry(poly, axis='x'):
    """Отражает многоугольник относительно указанной оси ('x' или 'y')."""
    if axis == 'x':
        return tuple((x, -y) for (x, y) in poly)
    elif axis == 'y':
        return tuple((-x, y) for (x, y) in poly)
    else:
        raise ValueError("Ось должна быть 'x' или 'y'")

def tr_homothety(k, poly, center=(0, 0)):
    """Применяет гомотетию с коэффициентом k относительно центра."""
    cx, cy = center
    return tuple((cx + k * (x - cx), cy + k * (y - cy)) for (x, y) in poly)

# 5. Вспомогательные функции для фильтров
def is_convex(poly):
    """Проверяет, является ли многоугольник выпуклым."""
    n = len(poly)
    if n < 3:
        return False
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
    signs = []
    for i in range(n):
        cp = cross_product(poly[i], poly[(i+1)%n], poly[(i+2)%n])
        if cp != 0:
            signs.append(cp > 0)
    return len(set(signs)) <= 1

def polygon_area(poly):
    """Вычисляет площадь многоугольника по формуле шнуровки."""
    n = len(poly)
    area = sum(poly[i][0] * poly[(i+1)%n][1] - poly[(i+1)%n][0] * poly[i][1]
               for i in range(n))
    return abs(area) / 2

def shortest_side(poly):
    """Находит длину кратчайшей стороны многоугольника."""
    n = len(poly)
    if n < 2:
        return 0
    return min(((poly[i][0] - poly[(i+1)%n][0])**2 +
                (poly[i][1] - poly[(i+1)%n][1])**2)**0.5 for i in range(n))

def point_inside_polygon(point, poly):
    """Проверяет, находится ли точка внутри многоугольника (алгоритм ray casting)."""
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Функции-фильтры
def make_flt_convex_polygon():
    """Фильтр для выпуклых многоугольников."""
    return is_convex

def make_flt_angle_point(point):
    """Фильтр для многоугольников, у которых есть вершина, равная заданной точке."""
    return lambda poly: any(p == point for p in poly)

def make_flt_square(max_area):
    """Фильтр для многоугольников с площадью меньше max_area."""
    return lambda poly: polygon_area(poly) < max_area

def make_flt_short_side(min_length):
    """Фильтр для многоугольников с кратчайшей стороной меньше min_length."""
    return lambda poly: shortest_side(poly) < min_length

def make_flt_point_inside(point):
    """Фильтр для выпуклых многоугольников, содержащих заданную точку."""
    return lambda poly: is_convex(poly) and point_inside_polygon(point, poly)

def make_flt_polygon_angles_inside(given_poly):
    """Фильтр для выпуклых многоугольников, содержащих любую вершину заданного многоугольника."""
    return lambda poly: is_convex(poly) and any(point_inside_polygon(p, poly)
                                               for p in given_poly)

# 7. Декораторы
def flt_convex_polygon(func):
    """Декоратор для фильтрации выпуклых многоугольников в первом аргументе."""
    def wrapper(poly_iter, *args, **kwargs):
        return func(filter(is_convex, poly_iter), *args, **kwargs)
    return wrapper

def make_decorator_flt_angle_point(point):
    def decorator(func):
        def wrapper(poly_iter, *args, **kwargs):
            return func(filter(make_flt_angle_point(point), poly_iter), *args, **kwargs)
        return wrapper
    return decorator

def translate_decorator(dx, dy):
    """Декоратор для применения сдвига к многоугольникам."""
    def decorator(func):
        def wrapper(poly_iter, *args, **kwargs):
            return func(map(lambda p: tr_translate(dx, dy, p), poly_iter), *args, **kwargs)
        return wrapper
    return decorator

# 8. Агрегирующие функции
def agr_origin_nearest(poly_seq):
    """Находит вершину, ближайшую к началу координат."""
    all_points = chain.from_iterable(poly_seq)
    return min(all_points, key=lambda p: p[0]**2 + p[1]**2, default=None)

def agr_max_side(poly_seq):
    """Находит максимальную длину стороны среди всех многоугольников."""
    return max((max(((p[i][0] - p[(i+1)%len(p)][0])**2 +
                     (p[i][1] - p[(i+1)%len(p)][1])**2)**0.5
                    for i in range(len(p))) for p in poly_seq), default=0)

def agr_min_area(poly_seq):
    """Находит минимальную площадь среди всех многоугольников."""
    return min(map(polygon_area, poly_seq), default=0)

def agr_perimeter(poly_seq):
    """Вычисляет общий периметр всех многоугольников."""
    return sum(map(lambda p: sum(((p[i][0] - p[(i+1)%len(p)][0])**2 +
                                  (p[i][1] - p[(i+1)%len(p)][1])**2)**0.5
                                 for i in range(len(p))), poly_seq), 0)

def agr_area(poly_seq):
    """Вычисляет общую площадь всех многоугольников."""
    return sum(map(polygon_area, poly_seq), 0)

# 9. Функции объединения (zipping)
def zip_polygons(*iterators):
    """Объединяет многоугольники из нескольких итераторов в один, соединяя их точки."""
    return map(lambda t: tuple(chain(*t)), zip(*iterators))

def count_2D(start1=0, start2=0, step1=1, step2=1):
    """Генерирует 2D-последовательность точек с началом в (start1, start2) и шагами (step1, step2)."""
    x, y = start1, start2
    while True:
        yield (x, y)
        x += step1
        y += step2

def zip_tuple(*iterators):
    """Объединяет кортежи из нескольких итераторов."""
    return zip(*iterators)

# Демонстрация функциональности

# 2. Генерация и визуализация семи фигур
print("Демонстрация 2: Семь фигур (2 прямоугольника, 3 треугольника, 2 шестиугольника)")
rects = itertools.islice(gen_rectangle(), 2)
tris = itertools.islice(gen_triangle(), 3)
hexes = itertools.islice(gen_hexagon(), 2)
seven_figures = itertools.chain(rects, tris, hexes)
visualize_polygons(seven_figures)

# 4. Создание и визуализация специфических конфигураций
print("Демонстрация 4a: Три параллельные ленты многоугольников")
theta = math.pi / 6  # 30 градусов
d = 2
dx = -d * math.sin(theta)
dy = d * math.cos(theta)
ribbon1 = map(lambda p: tr_rotate(theta, p), gen_rectangle())
ribbon2 = map(lambda p: tr_translate(dx, dy, tr_rotate(theta, p)), gen_rectangle())
ribbon3 = map(lambda p: tr_translate(2*dx, 2*dy, tr_rotate(theta, p)), gen_rectangle())
figures_4a = itertools.chain(itertools.islice(ribbon1, 5),
                             itertools.islice(ribbon2, 5),
                             itertools.islice(ribbon3, 5))
visualize_polygons(figures_4a)

print("Демонстрация 4b: Две пересекающиеся ленты")
ribbon1_4b = map(lambda p: tr_rotate(math.pi/6, p), gen_rectangle())
ribbon2_4b = map(lambda p: tr_translate(2, 2, tr_rotate(-math.pi/6, p)), gen_rectangle())
figures_4b = itertools.chain(itertools.islice(ribbon1_4b, 5),
                             itertools.islice(ribbon2_4b, 5))
visualize_polygons(figures_4b)

print("Демонстрация 4c: Две симметричные ленты треугольников")
ribbon1_4c = map(lambda p: tr_rotate(math.pi/6, p), gen_triangle())
ribbon2_4c = map(lambda p: tr_translate(0, 2, tr_rotate(-math.pi/6, p)), gen_triangle())
figures_4c = itertools.chain(itertools.islice(ribbon1_4c, 5),
                             itertools.islice(ribbon2_4c, 5))
visualize_polygons(figures_4c)

print("Демонстрация 4d: Четырехугольники в разных масштабах")
quads_4d = [tr_homothety(k, p) for k in [0.5, 1, 1.5, 2]
            for p in itertools.islice(gen_rectangle(), 1)]
visualize_polygons(quads_4d)

# 6. Фильтрация и визуализация
print("Демонстрация 6a: Фильтрация шести фигур из конфигурации 4a")
figures_6a = list(itertools.chain(
    itertools.islice(map(lambda p: tr_rotate(math.pi/6, p), gen_rectangle()), 5),
    itertools.islice(map(lambda p: tr_translate(dx, dy, tr_rotate(math.pi/6, p)), gen_rectangle()), 5),
    itertools.islice(map(lambda p: tr_translate(2*dx, 2*dy, tr_rotate(math.pi/6, p)), gen_rectangle()), 5)
))
filtered_6a = list(filter(make_flt_square(1.5), figures_6a))[:6]
visualize_polygons(filtered_6a)

print("Демонстрация 6b: Фильтрация фигур с кратчайшей стороной меньше значения")
figures_6b = [tr_homothety(k, p) for k in [i/2 for i in range(1, 16)]
              for p in itertools.islice(gen_rectangle(), 1)]
filtered_6b = list(filter(make_flt_short_side(0.8), figures_6b))[:4]
visualize_polygons(filtered_6b)

print("Демонстрация 6c: Фильтрация пересекающихся фигур")
figures_6c = list(itertools.chain(
    itertools.islice(map(lambda p: tr_rotate(math.pi/6, p), gen_rectangle()), 10),
    itertools.islice(map(lambda p: tr_translate(2, 2, tr_rotate(-math.pi/6, p)), gen_rectangle()), 10)
))
# Простой фильтр для демонстрации (можно улучшить для реальных пересечений)
filtered_6c = list(filter(make_flt_point_inside((2, 2)), figures_6c))[:4]
visualize_polygons(filtered_6c)

# 7. Демонстрация декораторов
print("Демонстрация 7: Применение декораторов")
@flt_convex_polygon
def process_polygons(poly_iter):
    visualize_polygons(poly_iter)

@translate_decorator(2, 2)
def process_shifted(poly_iter):
    visualize_polygons(poly_iter)

process_polygons(gen_rectangle())  # Визуализирует выпуклые многоугольники
process_shifted(gen_rectangle())   # Визуализирует сдвинутые прямоугольники

# 8. Применение агрегирующих функций
print("Демонстрация 8: Агрегирующие функции")
rects_8 = list(itertools.islice(gen_rectangle(), 5))
print(f"Ближайшая к началу координат вершина: {agr_origin_nearest(rects_8)}")
print(f"Максимальная сторона: {agr_max_side(rects_8)}")
print(f"Минимальная площадь: {agr_min_area(rects_8)}")
print(f"Общий периметр: {agr_perimeter(rects_8)}")
print(f"Общая площадь: {agr_area(rects_8)}")

# 9. Демонстрация функций объединения
print("Демонстрация 9a: zip_polygons")
p1_9a = [((1,1), (2,2), (3,1)), ((11,11), (12,12), (13,11))]
p2_9a = [((1,-1), (2,-2), (3,-1)), ((11,-11), (12,-12), (13,-11))]
result_9a = list(zip_polygons(p1_9a, p2_9a))
visualize_polygons(result_9a)

print("Демонстрация 9b: count_2D")
points_9b = list(itertools.islice(count_2D(0, 0, 1, 2), 5))
print(f"Результат count_2D: {points_9b}")

print("Демонстрация 9c: zip_tuple")
i1_9c = [(1,1), (2,2), (3,3), (4,4)]
i2_9c = [(2,2), (3,3), (4,4), (5,5)]
i3_9c = [(3,3), (4,4), (5,5), (6,6)]
result_9c = list(zip_tuple(i1_9c, i2_9c, i3_9c))
print(f"Результат zip_tuple: {result_9c}")
