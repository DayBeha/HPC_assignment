from JuliaSet import *


import pytest
@pytest.mark.parametrize('num1, num2, expected', [(1000, 300, 33219980),
(100, 300, 334236), (100, 100, 131532), (180, 300, 1076586), (180, 200, 1076586)])
def test_calc_pure_python(num1, num2, expected):
    assert sum(calc_pure_python(desired_width=num1, max_iterations=num2)) == expected
