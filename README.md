implementation:
1. Microfacet material
2. Multiple importance sampling
3. Malley's Method
4. reflection and refraction

Environment:

```
[Taichi] version 0.8.4, llvm 10.0.0, commit 895881b5, win, python 3.7.10
[Taichi] Starting on arch=cuda
```

Reference:
《Physically based rendering:from theory to implementation》

# path tracer based on 《PBRT》



![pbrt book](https://user-images.githubusercontent.com/68142480/141953284-a505688a-76f6-45f0-a234-011328d75799.jpg)




## 一.introduction to sampling theory

#### 1. what is sampling?

> impulse train：

![冲激链](https://user-images.githubusercontent.com/68142480/141953472-2f55ddc0-8e1e-4dc7-9bf6-25644282b8ce.png)

> sampling process corresponds to multiplying the function by a “impulse train” function, an infinite sum of equally spaced delta functions.


![采样](https://user-images.githubusercontent.com/68142480/141953566-38691937-b738-41de-95a8-3eeabea08212.png)

![采样的图示](https://user-images.githubusercontent.com/68142480/141953691-97b247fd-6a49-42e7-9b0f-c77771bc5cc8.png)


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


![cornell_box aliazing](https://user-images.githubusercontent.com/68142480/141953868-7e01127a-1cfa-4e10-b1bd-69c36de5a523.png)

> then we need anti-aliazing

```python
pos = camera_pos
ray_dir = ti.Vector([
            (2 * fov * (u + ti.random()) / resolution[1] - fov * resolution[0] / resolution[1] - 1e-5),
            2 * fov * (v + ti.random()) / resolution[1] - fov - 1e-5, -1.0
        ]).normalized()
```

![cornell_box](https://user-images.githubusercontent.com/68142480/141953907-cd387b79-c9d8-45aa-b846-9be19d328ad0.png)


## 二.sampling

#### Preview (CDF sampling technique)

> There are many techniques for generating random variates from a specified probability distribution such as the normal, exponential, or gamma distribution. However, one technique stands out because of its generality and simplicity: **<font color=red>the inverse CDF sampling technique</font>**.

![inverse CDF](https://user-images.githubusercontent.com/68142480/141953952-36ab9e40-5a30-4ece-9d2d-d7c3adde2e30.png)


#### 1. Uniformly Sampling a Hemisphere (multidimensional sampling technique)

![hemisphere sampling](https://user-images.githubusercontent.com/68142480/141953987-6bfd3ad8-4b69-4a46-8773-e6cf092304a8.png)


>  a uniform distribution means that the density function is a constant, so we know that p(x) = c

![半球均匀采样](https://user-images.githubusercontent.com/68142480/141954037-88876d6a-800f-4074-b5d1-fccb6463d5b6.png)

> so    p(ω) = 1/2*pi

> then    p(θ, φ) = sinθ/2*pi

![半球均匀采样-边缘密度](https://user-images.githubusercontent.com/68142480/141954428-4ed0a3fc-95a0-4430-b790-56fbe1422eab.png)

![半球均匀采样-条件密度](https://user-images.githubusercontent.com/68142480/141954411-364d9e06-b1ad-439d-8384-d98ebc535e11.png)

>  Notice that the density function for φ itself is uniform

>  then use the 1D inversion technique to sample each of these PDFs in turn 

![半球均匀采样4](https://user-images.githubusercontent.com/68142480/141954513-a1c0c849-5285-452f-ab2f-82e4f85df60d.png)

![半球均匀采样5](https://user-images.githubusercontent.com/68142480/141954524-b3117da6-e2d0-4b04-848e-79e3a438bc9d.png)

![半球均匀采样-Final](https://user-images.githubusercontent.com/68142480/141954536-c50160d5-bbdb-4744-8343-c1cd48922edc.png)


#### 2. sample area light

![sample light](https://user-images.githubusercontent.com/68142480/141954586-0c8814a1-41d2-45ea-929a-6ab06eec4768.png)

```python
def sample_area_light(hit_pos, pos_normal):
    # sampling inside the light area
    x = ti.random() * light_x_range + light_x_min_pos
    z = ti.random() * light_z_range + light_z_min_pos
    on_light_pos = ti.Vector([x, light_y_pos, z])
    return (on_light_pos - hit_pos).normalized()
```

![Sample Area Light](https://user-images.githubusercontent.com/68142480/141954636-8cb5c8b8-1342-48ba-90ce-609fbc8d9569.png)


#### 3. introduction to importance sampling

<font color=green>why we need importance sampling?</font>

> the Monte Carlo estimator  converges more quickly if the samples are taken from a distribution p(x) that is **<font color=red>similar</font>** to the function f(x) in the integrand.

![Monte Carlo estimator](https://user-images.githubusercontent.com/68142480/141954777-814df0d0-c9a1-4ad9-bb90-6057961f323b.png)

> 《PBRT》：We will not provide a rigorous proof of this fact but will instead present an informal and intuitive argument.

<font color=green>then we try to analyze the importance sampling method</font>

![蒙特卡洛积分](https://user-images.githubusercontent.com/68142480/141954814-5d642ed9-ec8f-45b1-a375-90124a528849.png)

we have three terms

- BRDF
- incident radiance ( infeasible )
- cosine term



#### 4. cosine-weighted sampling

![1637050619568](https://user-images.githubusercontent.com/68142480/141954912-39169e4f-ca8a-44cd-a174-a3466055e9c1.png)

![1637050643929](https://user-images.githubusercontent.com/68142480/141954948-16777d42-5973-4ecd-a802-492c11057d76.png)

![1637050669913](https://user-images.githubusercontent.com/68142480/141954976-17497aee-82be-4e60-89d9-cea7235e2869.png)

##### Malley's method

So, We could compute the marginal and conditional densities as before, **<font color=red>but instead we can use a technique known as *Malley’s* *method* to generate these cosine-weighted points</font>**.

![Malleys Method](https://user-images.githubusercontent.com/68142480/141955019-a3844f1f-1ee7-4d5a-8845-eb76d751251c.png)

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

![1637051228888](https://user-images.githubusercontent.com/68142480/141955091-44db4a32-b7e3-4845-ba83-747070fd50b4.png)

![1637051236921](https://user-images.githubusercontent.com/68142480/141955130-ebde2ad6-6094-4cf2-adfe-8a6e296f3757.png)

(2) projection

> To complete the **<font color=blue>(r,φ)=(sinθ,φ)⇒(θ,φ)</font>** transformation, we need the determinant of the Jacobian

![1637051213404](https://user-images.githubusercontent.com/68142480/141955204-4ce58f9a-c0c2-479b-8a35-86acf8050deb.png)

> Why![1637051266186](https://user-images.githubusercontent.com/68142480/141955276-b88dfae4-8159-4057-beac-b397a5d47763.png)

![1637051283108](https://user-images.githubusercontent.com/68142480/141955321-b371d62d-d3f6-4a7f-b51e-dabd1a8c2f67.png)



#### 5. multiple importance sampling

> BDPT only：

![1637051363951](https://user-images.githubusercontent.com/68142480/141955477-25f15b12-3465-4dad-b029-c628a616ad03.png)

> BDPT + MIS：

![1637051397587](https://user-images.githubusercontent.com/68142480/141955528-1c366532-e3e9-426c-9ac3-2d28a96fb7c1.png)

##### Why we need MIS?

![BSDF Only](https://user-images.githubusercontent.com/68142480/141955598-0903a130-05c4-4091-ba5a-903e96f33771.png)

![Light Only](https://user-images.githubusercontent.com/68142480/141955642-beafdbec-2adb-48c8-a6af-2160924b0703.png)



![1637051701718](https://user-images.githubusercontent.com/68142480/141955709-aa721a8f-69b0-42ee-83f7-527c0321ed2e.png)

![1637051712205](https://user-images.githubusercontent.com/68142480/141955756-c17d1da7-73cb-4890-b302-4f759c4d28b2.png)

![1637051724783](https://user-images.githubusercontent.com/68142480/141955796-e54e574b-7eb9-4072-8a4f-4d5939496c72.png)




> * balance heuristic
>
> ![1637051757112](https://user-images.githubusercontent.com/68142480/141955869-33d7599f-2748-4f7a-9542-793d6ab43d5d.png)
>
> * power heuristic (**Veach determined empirically that β=2 is a good value**.)
> 
> ![1637051764336](https://user-images.githubusercontent.com/68142480/141955906-de6cd036-d30d-4fea-aee5-cf41df987a26.png)
> 

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
