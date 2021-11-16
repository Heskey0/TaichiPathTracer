# path tracer based on 《PBRT》

<img src="C:\A\Media\录制\素材\pbrt book.jpg" alt="pbrt book" style="zoom: 67%;" />





## 一.introduction to sampling theory

#### 1. what is sampling?

> impulse train：

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\冲激链.png" alt="冲激链" style="zoom: 50%;" />

> sampling process corresponds to multiplying the function by a “impulse train” function, an infinite sum of equally spaced delta functions.

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\采样.png" alt="采样" style="zoom: 50%;" />

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\采样的图示.png" alt="采样的图示" style="zoom:60%;" />

> 《PBRT》A digital image is represented as a set of pixel values, typically aligned on a rectangular grid. When a digital image is displayed on a physical device, these values are used to determine the spectral power emitted by pixels on the display. 

> 《PBRT》the pixels that constitute an image are point samples of the image function at discrete points on the image plane.
>
> there is no “area” associated with a pixel.



> when sampling the film signal

```python
pos = camera_pos
ray_dir = ti.Vector([
            (2 * fov * (u) / resolution[1] - fov * resolution[0] / resolution[1] - 1e-5),
            2 * fov * (v) / resolution[1] - fov - 1e-5, -1.0
        ]).normalized()
```

![cornell_box aliazing](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\ForInstance\cornell_box aliazing.png)

> then we need anti-aliazing

```python
pos = camera_pos
ray_dir = ti.Vector([
            (2 * fov * (u + ti.random()) / resolution[1] - fov * resolution[0] / resolution[1] - 1e-5),
            2 * fov * (v + ti.random()) / resolution[1] - fov - 1e-5, -1.0
        ]).normalized()
```

![cornell_box](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\ForInstance\cornell_box.png)



## 二.sampling

#### Preview (CDF sampling technique)

> There are many techniques for generating random variates from a specified probability distribution such as the normal, exponential, or gamma distribution. However, one technique stands out because of its generality and simplicity: **<font color=red>the inverse CDF sampling technique</font>**.

![inverse CDF](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\Basic\inverse CDF.png)

#### 1. Uniformly Sampling a Hemisphere (multidimensional sampling technique)

![hemisphere sampling](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\hemisphere sampling.png)

>  a uniform distribution means that the density function is a constant, so we know that p(x) = c

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样.png" alt="半球均匀采样" style="zoom:60%;" />

> so    p(ω) = 1/2*pi

> then    p(θ, φ) = sinθ/2*pi

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样-边缘密度.png" alt="半球均匀采样-边缘密度" style="zoom:70%;" />

![半球均匀采样-条件密度](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样-条件密度.png)

>  Notice that the density function for φ itself is uniform

>  then use the 1D inversion technique to sample each of these PDFs in turn 

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样4.png" alt="半球均匀采样4" style="zoom:60%;" />

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样5.png" alt="半球均匀采样5" style="zoom:60%;" />

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\半球均匀采样-Final.png" alt="半球均匀采样-Final" style="zoom:60%;" />



#### 2. sample area light

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\sample light.png" alt="sample light" style="zoom: 67%;" />

```python
def sample_area_light(hit_pos, pos_normal):
    # sampling inside the light area
    x = ti.random() * light_x_range + light_x_min_pos
    z = ti.random() * light_z_range + light_z_min_pos
    on_light_pos = ti.Vector([x, light_y_pos, z])
    return (on_light_pos - hit_pos).normalized()
```

![Sample Area Light](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\ForInstance\Sample Area Light.png)

#### 3. introduction to importance sampling

<font color=green>why we need importance sampling?</font>

> the Monte Carlo estimator  converges more quickly if the samples are taken from a distribution p(x) that is **<font color=red>similar</font>** to the function f(x) in the integrand.

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\Monte Carlo estimator.png" alt="Monte Carlo estimator" style="zoom:60%;" />

> 《PBRT》：We will not provide a rigorous proof of this fact but will instead present an informal and intuitive argument.

<font color=green>then we try to analyze the importance sampling method</font>

<img src="C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\蒙特卡洛积分.png" alt="蒙特卡洛积分" style="zoom:60%;" />

we have three terms

- BRDF
- incident radiance ( infeasible )
- cosine term



#### 4. cosine-weighted sampling

![1637050619568](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637050619568.png)

![1637050643929](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637050643929.png)

![1637050669913](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637050669913.png)

##### Malley's method

So, We could compute the marginal and conditional densities as before, **<font color=red>but instead we can use a technique known as *Malley’s* *method* to generate these cosine-weighted points</font>**.

![Malleys Method](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\Basic\Malleys Method.png)

> 1. cosine term
>
> 2. 2D Sampling with Multidimensional Transformations
>
>    * **<font color=red>(1) sampling a unit disk</font>** (Concentric Mapping)
>
>    * **<font color=red>(2) project up to the unit hemisphere</font>** (cosine-weighted hemisphere sampling)
>
>      > 

(1) sampling a unit disk

![1637051228888](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051228888.png)

![1637051236921](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051236921.png)

(2) projection

> To complete the **<font color=blue>(r,φ)=(sinθ,φ)⇒(θ,φ)</font>** transformation, we need the determinant of the Jacobian

![1637051213404](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051213404.png)

> Why![1637051266186](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051266186.png)

![1637051283108](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051283108.png)



#### 5. multiple importance sampling

> BDPT only：

![1637051363951](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051363951.png)

> BDPT + MIS：

![1637051397587](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051397587.png)

##### Why we need MIS?

![BSDF Only](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\ForInstance\BSDF Only.png)

![Light Only](C:\Users\34957\Desktop\PornHub\FMA讲课\srcImages\ForInstance\Light Only.png)



![1637051701718](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051701718.png)

![1637051712205](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051712205.png)

![1637051724783](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051724783.png)



> * balance heuristic
>
> ![1637051757112](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051757112.png)
>
> * power heuristic (**Veach determined empirically that β=2 is a good value**.)
>
> ![1637051764336](C:\Users\34957\AppData\Roaming\Typora\typora-user-images\1637051764336.png)

```python
//Compute heuristic
def mis_power_heuristic(pf, pg):
    # Assume 1 sample for each distribution
    f = pf ** 2
    g = pg ** 2
    return f / (f + g)
    # return 1
```

```python
//combine
@ti.func
def sample_light_and_cosineWeighted(hit_pos, hit_normal):
    cosine_by_pdf = ti.Vector([0.0, 0.0, 0.0])

    light_pdf, cosineWeighted_pdf = 0.0, 0.0

    # sample area light => dir, light_pdf; then dir => lambertian_pdf; then mis
    light_dir = sample_area_light(hit_pos, hit_normal)
    if light_dir.dot(hit_normal) > 0:
        light_pdf = compute_area_light_pdf(hit_pos, light_dir)
        cosineWeighted_pdf = compute_cosineWeighted_pdf(hit_normal, light_dir)
        if light_pdf > 0 and cosineWeighted_pdf > 0:
            l_visible = visible_to_light(hit_pos, light_dir)
            if l_visible:
                heuristic = compute_heuristic(light_pdf, cosineWeighted_pdf)
                DoN = dot_or_zero(light_dir, hit_normal)
                cosine_by_pdf += heuristic * DoN / light_pdf

    # sample cosine weighted => dir, lambertian_pdf; then dir => light_pdf; then mis
    cosineWeighted_dir = cosine_weighted_sampling(hit_normal)
    cosineWeighted_pdf = compute_cosineWeighted_pdf(hit_normal, cosineWeighted_dir)
    light_pdf = compute_area_light_pdf(hit_pos, cosineWeighted_dir)
    if visible_to_light(hit_pos, cosineWeighted_dir):
        heuristic = compute_heuristic(cosineWeighted_pdf, light_pdf)
        DoN = dot_or_zero(cosineWeighted_dir, hit_normal)
        cosine_by_pdf += heuristic * DoN / cosineWeighted_pdf

    # direct_li = mis_weight * cosine / pdf
    return cosine_by_pdf
```
