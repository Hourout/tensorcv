import tensorflow as tf

__all__ = ['softshrink']

def softshrink(x, delta=0.5, name=None):
    """Applies the soft shrinkage function elementwise:
    
    if x > delta, output = x - delta;
    if x < -delta, output = x + delta;
    otherwise, output = 0
    Args:
        x: A Tensor with type float32 or float64.
        delta: the value for the Softshrink formulation. Default: 0.5
        name: A name for the operation (optional).
    Returns:
        A Tensor with the same type as x.
    """
    def f1(): return x+delta
    def f2(): return x-delta
    def f3(): return 0.
    return tf.case({tf.less(x, -delta): f1, 
                    tf.greater(x, delta): f2},
                   default=f3, exclusive=True)
