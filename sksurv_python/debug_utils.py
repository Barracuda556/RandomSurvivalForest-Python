# sksurv_python/debug_utils.py
import numpy as np

def compare_predictions(original_pred, python_pred, X_test=None, tolerance=1e-3):
    """Сравнить предсказания оригинальной и Python реализации."""
    print("=" * 80)
    print("СРАВНЕНИЕ ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    print(f"\nОригинальные предсказания (первые 10):")
    print(original_pred[:10])
    
    print(f"\nPython предсказания (первые 10):")
    print(python_pred[:10])
    
    print(f"\nРазличия (первые 10):")
    diff = original_pred - python_pred
    print(diff[:10])
    
    print(f"\nСтатистика различий:")
    print(f"  Среднее абсолютное различие: {np.mean(np.abs(diff)):.6f}")
    print(f"  Максимальное абсолютное различие: {np.max(np.abs(diff)):.6f}")
    print(f"  Минимальное абсолютное различие: {np.min(np.abs(diff)):.6f}")
    print(f"  Стандартное отклонение различий: {np.std(diff):.6f}")
    
    # Проверить, насколько похожи распределения
    print(f"\nКоэффициент корреляции Пирсона: {np.corrcoef(original_pred, python_pred)[0,1]:.6f}")
    
    if np.allclose(original_pred, python_pred, atol=tolerance):
        print(f"\n✓ Предсказания совпадают с точностью {tolerance}")
    else:
        print(f"\n✗ Предсказания НЕ совпадают с точностью {tolerance}")
    
    return diff

def debug_tree_structure(tree, node_id=0, depth=0):
    """Отладить структуру дерева."""
    if depth == 0:
        print("\n" + "=" * 80)
        print("СТРУКТУРА ДЕРЕВА")
        print("=" * 80)
    
    node = tree.nodes[node_id]
    indent = "  " * depth
    
    if node.left_child == -1 and node.right_child == -1:
        # Лист
        print(f"{indent}Лист {node_id}: samples={node.n_node_samples}, impurity={node.impurity:.4f}")
        
        # Показать значения
        if hasattr(tree, 'value') and tree.value is not None:
            value_start = node_id * tree.value_stride
            value_end = (node_id + 1) * tree.value_stride
            values = tree.value[value_start:value_end]
            if len(values) > 0:
                print(f"{indent}  Значения (первые 5): {values[:5]}")
                print(f"{indent}  Сумма значений: {np.sum(values):.4f}")
    else:
        # Внутренний узел
        print(f"{indent}Узел {node_id}: feature={node.feature}, threshold={node.threshold:.4f}, "
              f"samples={node.n_node_samples}, impurity={node.impurity:.4f}")
        
        # Рекурсивно обойти детей
        debug_tree_structure(tree, node.left_child, depth + 1)
        debug_tree_structure(tree, node.right_child, depth + 1)

def compare_tree_structures(tree1, tree2, max_nodes=20):
    """Сравнить структуры двух деревьев."""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ СТРУКТУР ДЕРЕВЬЕВ")
    print("=" * 80)
    
    n_nodes1 = tree1.node_count
    n_nodes2 = tree2.node_count
    
    print(f"\nДерево 1: {n_nodes1} узлов")
    print(f"Дерево 2: {n_nodes2} узлов")
    
    # Сравнить первые несколько узлов
    print("\nСравнение первых узлов:")
    print(f"{'Node':<6} {'Feature1':<10} {'Threshold1':<12} {'Feature2':<10} {'Threshold2':<12}")
    print("-" * 60)
    
    for i in range(min(max_nodes, n_nodes1, n_nodes2)):
        node1 = tree1.nodes[i]
        node2 = tree2.nodes[i]
        
        feature1 = node1.feature if node1.feature != -2 else "Leaf"
        feature2 = node2.feature if node2.feature != -2 else "Leaf"
        
        threshold1 = f"{node1.threshold:.4f}" if node1.threshold != -2 else "Leaf"
        threshold2 = f"{node2.threshold:.4f}" if node2.threshold != -2 else "Leaf"
        
        print(f"{i:<6} {str(feature1):<10} {threshold1:<12} {str(feature2):<10} {threshold2:<12}")

def check_criterion_values(criterion, node_id, X, y, sample_weight):
    """Проверить значения критерия для узла."""
    print(f"\nПроверка критерия для узла {node_id}:")
    
    # Получить значение узла из дерева
    if hasattr(criterion, 'node_value'):
        dest = np.zeros(criterion.n_outputs * criterion.n_unique_times)
        criterion.node_value(dest)
        print(f"  Значение узла (первые 10): {dest[:10]}")
        print(f"  Сумма значений: {np.sum(dest):.4f}")
    
    # Проверить статистики риска
    if hasattr(criterion, 'riskset_total'):
        print(f"\n  Статистики риска (первые 5 времен):")
        for i in range(min(5, criterion.n_unique_times)):
            at_risk, events = criterion.riskset_total.at(i)
            print(f"    Время {i}: at_risk={at_risk:.1f}, events={events:.1f}")
