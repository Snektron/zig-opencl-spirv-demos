const std = @import("std");
const cl = @import("opencl");
const common = @import("common.zig");

pub const std_options = common.std_options;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const platform, const device = try common.parseOptions(allocator);

    const context = try cl.createContext(&.{device}, .{ .platform = platform });
    defer context.release();

    const queue = try cl.createCommandQueue(context, device, .{ .profiling = true });
    defer queue.release();

    const program = try common.buildSpvProgram(allocator, context, device, @embedFile("saxpy-kernel"));
    defer program.release();

    const kernel = try cl.createKernel(program, "saxpy");
    defer kernel.release();

    std.log.debug("generating inputs...", .{});

    const size = 128 * 1024 * 1024;

    const y, const x = blk: {
        const y = try allocator.alloc(f32, size);
        const x = try allocator.alloc(f32, size);
        for (x, 0..) |*value, i| value.* = @floatFromInt(i);
        for (y, 0..) |*value, i| value.* = @floatFromInt(i + 1000);
        break :blk .{ y, x };
    };

    const results = try allocator.alloc(f32, size);

    const a: f32 = 123;

    const d_y = try cl.createBufferWithData(f32, context, .{ .read_write = true }, y);
    const d_x = try cl.createBufferWithData(f32, context, .{ .read_only = true }, x);

    std.log.debug("launching kernel with {} inputs per array", .{size});

    try kernel.setArg(@TypeOf(d_y), 0, d_y);
    try kernel.setArg(@TypeOf(d_x), 1, d_x);
    try kernel.setArg(f32, 2, a);

    const saxpy_complete = try queue.enqueueNDRangeKernel(
        kernel,
        null,
        &.{size},
        &.{256},
        &.{},
    );
    defer saxpy_complete.release();

    const read_complete = try queue.enqueueReadBuffer(
        f32,
        d_y,
        false,
        0,
        results,
        &.{saxpy_complete},
    );
    defer read_complete.release();

    try cl.waitForEvents(&.{read_complete});

    const start = try saxpy_complete.commandStartTime();
    const stop = try saxpy_complete.commandEndTime();
    const runtime = stop - start;
    const tput = size * @sizeOf(f32) * std.time.ns_per_s / runtime;
    std.log.info("kernel took {d} us, {:2}/s", .{runtime / std.time.ns_per_us, std.fmt.fmtIntSizeBin(tput)});

    std.log.debug("checking results...", .{});

    // Compute reference results on host
    for (y, x) |*yi, xi| {
        yi.* += xi * a;
    }

    // Check if the results are close.
    // y = y + a * x is 2 operations of 0.5 ulp each,
    // multiply by 2 for host and device side error.
    const max_error = std.math.floatEps(f32) * 2 * 2;
    for (results, y, 0..) |ri, yi, i| {
        if (!std.math.approxEqRel(f32, ri, yi, max_error)) {
            common.fail("invalid result at index {}: expected = {d}, actual = {d}", .{ i, yi, ri });
        }
    }

    std.log.info("results ok", .{});
}
