#!/usr/bin/env python

import theano
import theano.tensor as T

x = T.scalar()
x

y = 3 * (x**2) + 1
y
print y



