const std = @import("std");

const DemoBuild = struct {
    const Kernel = struct {
        name: []const u8,
        spv_path: std.Build.LazyPath,
    };

    b: *std.Build,
    host_target: std.Build.ResolvedTarget,
    device_target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    opencl: *std.Build.Module,

    fn addKernel(self: DemoBuild, name: []const u8, root_source_file: []const u8) Kernel {
        const kernel = self.b.addStaticLibrary(.{
            .name = "saxpy-kernel",
            .root_source_file = self.b.path(root_source_file),
            .target = self.device_target,
            .optimize = self.optimize,
            .use_llvm = false,
            .use_lld = false,
        });

        const dis = self.b.addSystemCommand(&.{"spirv-dis"});
        dis.addFileArg(kernel.getEmittedBin());

        const dis_step = self.b.step(
            self.b.fmt("dis-{s}", .{name}),
            self.b.fmt("Disassemble the {s} kernel ({s})", .{name, root_source_file}),
        );
        dis_step.dependOn(&dis.step);

        return .{
            .name = self.b.dupe(name),
            .spv_path = kernel.getEmittedBin(),
        };
    }

    fn addSpvasmKernel(self: DemoBuild, name: []const u8, root_source_file: []const u8) Kernel {
        const as = self.b.addSystemCommand(&.{"spirv-as", "--target-env", "spv1.4"});
        as.addFileArg(self.b.path(root_source_file));
        as.addArg("-o");
        const spv = as.addOutputFileArg(self.b.fmt("{s}.spv", .{name}));
        return .{ .name = name, .spv_path = spv };
    }

    fn addDemo(self: DemoBuild, name: []const u8, root_source_file: []const u8, kernels: []const Kernel) void {
        const exe = self.b.addExecutable(.{
            .name = name,
            .root_source_file = self.b.path(root_source_file),
            .target = self.host_target,
            .optimize = self.optimize,
            .link_libc = true,
        });
        exe.root_module.addImport("opencl", self.opencl);
        for (kernels) |kernel| {
            exe.root_module.addAnonymousImport(kernel.name, .{
                .root_source_file = kernel.spv_path,
            });
        }
        self.b.installArtifact(exe);

        const run_cmd = self.b.addRunArtifact(exe);
        run_cmd.step.dependOn(self.b.getInstallStep());
        if (self.b.args) |args| {
            run_cmd.addArgs(args);
        }

        const run_step = self.b.step(
            self.b.fmt("run-{s}", .{name}),
            self.b.fmt("Run the {s} demo", .{name}),
        );
        run_step.dependOn(&run_cmd.step);
    }
};

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const device_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv64,
        .os_tag = .opencl,
        .abi = .none,
        .cpu_features_add = std.Target.spirv.featureSet(&.{
            .int64,
            .int16,
            .int8,
            .float64,
            .float16,
            .vector16,
            .generic_pointer,
        }),
    });

    const opencl = b.dependency("opencl", .{
        .target = host_target,
        .optimize = optimize,
    }).module("opencl");

    const db = DemoBuild{
        .b = b,
        .host_target = host_target,
        .device_target = device_target,
        .optimize = optimize,
        .opencl = opencl,
    };

    db.addDemo("saxpy", "src/saxpy.zig", &.{
        db.addKernel("saxpy-kernel", "src/kernels/saxpy.zig"),
    });

    db.addDemo("reduce", "src/reduce.zig", &.{
        db.addKernel("reduce-kernel", "src/kernels/reduce.zig"),
    });
}
