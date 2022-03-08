#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define GRP_SIZE 16
#define PARAM_M 2.5
#define PARAM_W 1
#define LN_10 2.3025850929940459

<@Type> q_sol(<@Type> a, <@Type> b) 
{
    return 2.0 / (1.0 / (a * a) + 1.0 / (b * b));
}

// Absorption function
// Optimal parameters (probably): m = 2.5, W = 0.001
// dx - defines step. Step can differ depending on which direction absorption is applied to.
// l - field length along that direction excluding PML layers
<@Type> sigma(<@Type> x, <@Type> dx) 
{
    // d - PML layer width. 
    <@Type> d = <@PMLSize> * dx;
    return ((PARAM_M + 1) * PARAM_W * LN_10 / d) * pow(fabs(x / d), PARAM_M);
}
// Central Flux Limiter
<@Type> CFL(<@Type> r) 
{
    return max(0.0,
        min(2 * r, min(0.5 * (1 + r), 2.0)));
}

// mu = dt / dx; Courant number
<@Type> Absorption(<@Type> un1, <@Type> u0, <@Type> u1, <@Type> u2, <@Type> x, <@Type> dx, <@Type> dt, <@Type> localSpeed)
{
    <@Type> du0 = un1 - u0;
    <@Type> du1 = u0 - u1;
    <@Type> du2 = u1 - u2;

    <@Type> mu = dt / dx * localSpeed;
    <@Type> ratio = mu;
    if (du0 != 0 && du1 != 0)
    {
        ratio = mu + mu * (1 - mu) * 0.5 * (CFL(du1 / du0) * du0 / du1 - CFL(du2 / du1));
    }

    <@Type> result = u0 + (u1 - u0) * ratio - dt * sigma(x, dx) * u0;
    return result;
}

// Assuming the solution is performed on the fields WITH PML layers with size <@PMLSize>

__attribute__((reqd_work_group_size(GRP_SIZE, 1, 1)))
void kernel Step(
    global <@Type> * p2, // Write here
    global <@Type> * p1, // Previous step
    global <@Type> * p0, // One before previous step
    global const <@Type>* v) // Velocity field
{
    // Figure out indexes

    uint globalSizeX = <@SizeX> + 2 * <@PMLSize>;
    uint globalSizeY = <@SizeY> + 2 * <@PMLSize>;
    uint globalLimitX = globalSizeX - 1;
    uint globalLimitY = globalSizeY - 1;
    uint globalSize = globalSizeX * globalSizeY;

    int i = get_global_id(0);

    if (i >= globalSize)
        return;


    int x = i % globalSizeX;
    int y = i / globalSizeX;

    if (x <= 0 || x >= globalLimitX || y <= 0 || y >= globalLimitY)
    {
        p2[i] = 0;
        return;
    }
    // Inversed central cross condition to exclude corners
    if (  !((x > <@PMLSize> && x < globalLimitX - <@PMLSize>) || (y > <@PMLSize> && y < globalLimitY - <@PMLSize>))  )
    {
        p2[i] = 0;
        return;
    }

    //p2[i] = p1[i];
    //return;

    // Evaluate constants (that will be optimized during compilation anyways)

    <@Type> dx = <@DimX> / <@SizeX>;
    <@Type> dy = <@DimY> / <@SizeY>;
    <@Type> dx2 = dx * dx;
    <@Type> dy2 = dy * dy;
    <@Type> dt = <@T>;
    <@Type> dt2 = dt * dt;

    // SEPARATE BY PML REGIONS
    
    if (x <= <@PMLSize>) // Left side
    {
        p2[i] = Absorption(
            p1[i - 1], p1[i], p1[i + 1], p1[i + 2], 
            (<@PMLSize> - x) * dx, 
            dx, dt, v[0 + (y - <@PMLSize>) * <@SizeX>]);
    }
    else if (x >= globalLimitX - <@PMLSize>) // Right side
    {
        p2[i] = Absorption(
            p1[i + 1], p1[i], p1[i - 1], p1[i - 2], 
            (x - (globalLimitX - <@PMLSize>)) * dx,
            dx, dt, v[(<@SizeX> - 1) + (y - <@PMLSize>) * <@SizeX>]);
    }
    else if (y <= <@PMLSize>) // Top CAUSES SYSTEM CRASH IF EQUAL CONDITION IS REMOVED
    {
        p2[i] = Absorption(
            p1[i - globalSizeX], p1[i], p1[i + globalSizeX], p1[i + 2 * globalSizeX],
            (<@PMLSize> - y) * dy, 
            dy, dt, v[(x - <@PMLSize>) + 0 * <@SizeX>]);
    }
    else if (y >= globalLimitY - <@PMLSize>) // Bottom Contains anomaly, reflects at random
    {
        p2[i] = Absorption(
            p1[i + globalSizeX], p1[i], p1[i - globalSizeX], p1[i - 2 * globalSizeX],
            (y - (globalLimitY - <@PMLSize>)) * dy,
            dy, dt, v[(x - <@PMLSize>) + (<@SizeY> - 1) * <@SizeX>]);
    }
    else
    {
        // Evaluate them tricky velocity fields
        // Velocity field is unaffected by PML extension
        <@Type> q_l = 1, q_r = 1, q_t = 1, q_b = 1;
        uint vx = min(max(x - (<@PMLSize>), 0), <@SizeX> - 1);
        uint vy = min(max(y - (<@PMLSize>), 0), <@SizeY> - 1);
        uint v_i = vx + vy * <@SizeX>; 
        q_l = q_sol(v[v_i], v[v_i - 1]);
        q_r = q_sol(v[v_i], v[v_i + 1]);
        q_t = q_sol(v[v_i], v[v_i - <@SizeX>]);
        q_b = q_sol(v[v_i], v[v_i + <@SizeX>]);

        p2[i] = ((-q_r - q_l) / dx2 * dt2 + (-q_t - q_b) / dy2 * dt2 + 2.0) * p1[i] // Central term
            + q_l / dx2 * dt2 * p1[i - 1] // Left term
            + q_r / dx2 * dt2 * p1[i + 1] // Right term
            + q_t / dy2 * dt2 * p1[i - globalSizeX] // Top term
            + q_b / dy2 * dt2 * p1[i + globalSizeX] // Bottom term
            - p0[i]; // Lag term
    }
}