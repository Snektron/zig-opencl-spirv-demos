export fn saxpy(y: [*]addrspace(.global) f32, x: [*]addrspace(.global) const f32, a: f32) callconv(.Kernel) void {
    const gid = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    y[gid] += x[gid] * a;
}
