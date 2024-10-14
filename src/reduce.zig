const std = @import("std");
const cl = @import("opencl");
const common = @import("common.zig");
const reduce = @import("kernels/reduce.zig");

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

    const program = try common.buildSpvProgram(allocator, context, device, @embedFile("reduce-kernel"));
    defer program.release();

    const kernel = try cl.createKernel(program, "reduce");
    defer kernel.release();

    std.log.debug("generating inputs...", .{});

    const size = 256 * 1024 * 1024;

    const input = blk: {
        const values = try allocator.alloc(f32, size);
        var rng = std.Random.DefaultPrng.init(0);
        const random = rng.random();
        for (values) |*value| value.* = random.float(f32);
        break :blk values;
    };

    var d_input = try cl.createBufferWithData(f32, context, .{ .read_write = true }, input);
    var d_output = try cl.createBuffer(f32, context, .{ .read_write = true }, input.len);
    defer d_input.release();
    defer d_output.release();

    var maybe_event: ?cl.Event = null;
    var first_event: ?cl.Event = null;
    var remaining_size: usize = input.len;
    while (remaining_size != 1) {
        const blocks = std.math.divCeil(usize, remaining_size, reduce.items_per_block) catch unreachable;
        const valid_in_last_block = remaining_size % reduce.items_per_block;
        std.log.debug("reducing {} items over {} block(s)", .{ remaining_size, blocks });

        try kernel.setArg(@TypeOf(d_input), 0, d_input);
        try kernel.setArg(@TypeOf(d_output), 1, d_output);
        try kernel.setArg(u32, 2, @intCast(blocks - 1));
        try kernel.setArg(u32, 3, @intCast(valid_in_last_block));

        maybe_event = try queue.enqueueNDRangeKernel(
            kernel,
            null,
            &.{blocks * reduce.block_dim},
            &.{reduce.block_dim},
            if (maybe_event) |event| &.{event} else &.{},
        );

        if (first_event == null) {
            first_event = maybe_event.?;
        }

        const d_tmp = d_input;
        d_input = d_output;
        d_output = d_tmp;

        remaining_size = blocks;
    }

    var result: f32 = undefined;
    const read_complete = try queue.enqueueReadBuffer(
        f32,
        d_input,
        false,
        0,
        @as(*[1]f32, &result),
        if (maybe_event) |event| &.{event} else &.{},
    );
    defer read_complete.release();
    try cl.waitForEvents(&.{read_complete});

    if (maybe_event) |last_event| {
        const start = try first_event.?.commandStartTime();
        const stop = try last_event.commandEndTime();
        const runtime = stop - start;
        const tput = input.len * @sizeOf(f32) * std.time.ns_per_s / runtime;
        std.log.info("reduction took {d:.2} us, {:.2}/s", .{runtime / std.time.ns_per_us, std.fmt.fmtIntSizeBin(tput)});
    }

    // input.len * random in [0, 1) yields an average of input.len * 0.5
    const expected = input.len / 2;
    std.log.debug("result: {d}, expected: {d}", .{result, expected});
}

