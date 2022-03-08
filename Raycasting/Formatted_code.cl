#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define GRP_SIZE 16
#define PARAM_M 2.5
#define PARAM_W 1
#define LN_10 2.3025850929940459

double q_sol(double a, double b) 
{
    return 2.0 / (1.0 / (a * a) + 1.0 / (b * b));
}

// Absorption function
// Optimal parameters (probably): m = 2.5, W = 0.001
// dx - defines step. Step can differ depending on which direction absorption is applied to.
// l - field length along that direction excluding PML layers
double sigma(double x, double dx) 
{
    // d - PML layer width. 
    double d = 30 * dx;
    return ((PARAM_M + 1) * PARAM_W * LN_10 / d) * pow(fabs(x / d), PARAM_M);
}
// Central Flux Limiter
double CFL(double r) 
{
    return max(0.0,
        min(2 * r, min(0.5 * (1 + r), 2.0)));
}

// mu = dt / dx; Courant number
double Absorption(double un1, double u0, double u1, double u2, double x, double dx, double dt, double localSpeed)
{
    double du0 = un1 - u0;
    double du1 = u0 - u1;
    double du2 = u1 - u2;

    double mu = dt / dx * localSpeed;
    double ratio = mu;
    if (du0 != 0 && du1 != 0)
    {
        ratio = mu + mu * (1 - mu) * 0.5 * (CFL(du1 / du0) * du0 / du1 - CFL(du2 / du1));
    }

    double result = u0 + (u1 - u0) * ratio - dt * sigma(x, dx) * u0;
    return result;
}

// Assuming the solution is performed on the fields WITH PML layers with size 30

__attribute__((reqd_work_group_size(GRP_SIZE, 1, 1)))
void kernel Step(
    global double * p2, // Write here
    global double * p1, // Previous step
    global double * p0, // One before previous step
    global const double* v) // Velocity field
{
    // Figure out indexes

    uint globalSizeX = 1401 + 2 * 30;
    uint globalSizeY = 700 + 2 * 30;
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
    if (  !((x > 30 && x < globalLimitX - 30) || (y > 30 && y < globalLimitY - 30))  )
    {
        p2[i] = 0;
        return;
    }

    //p2[i] = p1[i];
    //return;

    // Evaluate constants (that will be optimized during compilation anyways)

    double dx = 7005.000000 / 1401;
    double dy = 3500.000000 / 700;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double dt = 0.000050;
    double dt2 = dt * dt;

    // SEPARATE BY PML REGIONS
    
    if (x <= 30) // Left side
    {
        p2[i] = Absorption(
            p1[i - 1], p1[i], p1[i + 1], p1[i + 2], 
            (30 - x) * dx, 
            dx, dt, v[0 + (y - 30) * 1401]);
    }
    else if (x >= globalLimitX - 30) // Right side
    {
        p2[i] = Absorption(
            p1[i + 1], p1[i], p1[i - 1], p1[i - 2], 
            (x - (globalLimitX - 30)) * dx,
            dx, dt, v[(1401 - 1) + (y - 30) * 1401]);
    }
    else if (y <= 30) // Top CAUSES SYSTEM CRASH IF EQUAL CONDITION IS REMOVED
    {
        p2[i] = Absorption(
            p1[i - globalSizeX], p1[i], p1[i + globalSizeX], p1[i + 2 * globalSizeX],
            (30 - y) * dy, 
            dy, dt, v[(x - 30) + 0 * 1401]);
    }
    else if (y >= globalLimitY - 30) // Bottom Contains anomaly, reflects at random
    {
        p2[i] = Absorption(
            p1[i + globalSizeX], p1[i], p1[i - globalSizeX], p1[i - 2 * globalSizeX],
            (y - (globalLimitY - 30)) * dy,
            dy, dt, v[(x - 30) + (700 - 1) * 1401]);
    }
    else
    {
        // Evaluate them tricky velocity fields
        // Velocity field is unaffected by PML extension
        double q_l = 1, q_r = 1, q_t = 1, q_b = 1;
        uint vx = min(max(x - (30), 0), 1401 - 1);
        uint vy = min(max(y - (30), 0), 700 - 1);
        uint v_i = vx + vy * 1401; 
        q_l = q_sol(v[v_i], v[v_i - 1]);
        q_r = q_sol(v[v_i], v[v_i + 1]);
        q_t = q_sol(v[v_i], v[v_i - 1401]);
        q_b = q_sol(v[v_i], v[v_i + 1401]);

        p2[i] = ((-q_r - q_l) / dx2 * dt2 + (-q_t - q_b) / dy2 * dt2 + 2.0) * p1[i] // Central term
            + q_l / dx2 * dt2 * p1[i - 1] // Left term
            + q_r / dx2 * dt2 * p1[i + 1] // Right term
            + q_t / dy2 * dt2 * p1[i - globalSizeX] // Top term
            + q_b / dy2 * dt2 * p1[i + globalSizeX] // Bottom term
            - p0[i]; // Lag term
    }
}