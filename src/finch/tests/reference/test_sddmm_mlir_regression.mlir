Compiling MLIR code:
module {

  func.func @scansearch(
      %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
        %1 = arith.constant 1 : index
        %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
            %plt = arith.cmpi slt, %p, %hi : index
            %cond = scf.if %plt -> (i1) {
                %ap = memref.load %arr[%p] : memref<?xindex>
                %al = arith.cmpi slt, %ap, %x : index
                scf.yield %al : i1
            } else {
                %f = arith.constant false
                scf.yield %f : i1
            }
            scf.condition(%cond) %d, %p : index, index
        } do {
        ^bb0(%d: index, %p: index):
            %d2 = arith.shli %d, %1 : index
            %p2 = arith.addi %p, %d2 : index
            scf.yield %d2, %p2 : index, index
        }
        %lo1 = arith.subi %g#1, %g#0 : index
        %minp = arith.minsi %g#1, %hi : index
        %hi1 = arith.addi %minp, %1 : index
        %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
            %hm1 = arith.subi %h, %1 : index
            %go = arith.cmpi slt, %l, %hm1 : index
            scf.condition(%go) %l, %h : index, index
        } do {
        ^bb0(%l: index, %h: index):
            %diff = arith.subi %h, %l : index
            %half = arith.shrsi %diff, %1 : index
            %m = arith.addi %l, %half : index
            %am = memref.load %arr[%m] : memref<?xindex>
            %al = arith.cmpi slt, %am, %x : index
            %l2, %h2 = scf.if %al -> (index, index) {
                scf.yield %m, %h : index, index
            } else {
                scf.yield %l, %m : index, index
            }
            scf.yield %l2, %h2 : index, index
        }
        return %b#1 : index
    }

  func.func @main(%_A_19: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_20: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_21: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %__A_59_72: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_9: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_19[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_19[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_19[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_19[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_20[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_20[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = arith.index_cast %v_13 : i64 to index
    %v_15 = llvm.extractvalue %_A_20[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_21[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = builtin.unrealized_conversion_cast %v_17 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_19 = llvm.extractvalue %_A_21[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_21[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_22 = arith.index_cast %v_21 : i64 to index
    %v_23 = llvm.extractvalue %__A_59_72[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = builtin.unrealized_conversion_cast %v_23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_25 = llvm.extractvalue %__A_59_72[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = llvm.extractvalue %__A_59_72[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_28 = arith.index_cast %v_27 : i64 to index
    %v_29 = llvm.extractvalue %_A_9[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_30 = builtin.unrealized_conversion_cast %v_29 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_31 = llvm.extractvalue %_A_9[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_32 = arith.index_cast %v_31 : i64 to index
    %v_33 = llvm.extractvalue %_A_9[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_34 = arith.index_cast %v_33 : i64 to index
    %v_35 = arith.constant 0 : index
    %v_36 = memref.dim %v_24, %v_35 : memref<?xf64>
    %v_37 = arith.constant 1 : index
    scf.for %v_38 = %v_35 to %v_36 step %v_37 {
      %v_39 = arith.constant 0.0 : f64
      memref.store %v_39, %v_24[%v_38] : memref<?xf64>
    }
    scf.for %v_40 = %v_35 to %v_14 step %v_37 {
      %v_41 = llvm.extractvalue %__A_59_72[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_42 = arith.index_cast %v_41 : i64 to index
      %v_43 = arith.muli %v_42, %v_40 : index
      %v_44 = arith.addi %v_35, %v_43 : index
      %v_45 = llvm.extractvalue %_A_20[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_46 = arith.index_cast %v_45 : i64 to index
      %v_47 = arith.muli %v_46, %v_40 : index
      %v_48 = arith.addi %v_35, %v_47 : index
      scf.for %v_49 = %v_35 to %v_16 step %v_37 {
        %v_50 = llvm.extractvalue %_A_20[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_51 = arith.index_cast %v_50 : i64 to index
        %v_52 = arith.muli %v_51, %v_49 : index
        %v_53 = arith.addi %v_48, %v_52 : index
        %v_54 = llvm.extractvalue %_A_21[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_55 = arith.index_cast %v_54 : i64 to index
        %v_56 = arith.muli %v_55, %v_49 : index
        %v_57 = arith.addi %v_35, %v_56 : index
        scf.for %v_58 = %v_35 to %v_22 step %v_37 {
          %v_59 = llvm.extractvalue %__A_59_72[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_60 = arith.index_cast %v_59 : i64 to index
          %v_61 = arith.muli %v_60, %v_58 : index
          %v_62 = arith.addi %v_44, %v_61 : index
          %v_63 = llvm.extractvalue %_A_21[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_64 = arith.index_cast %v_63 : i64 to index
          %v_65 = arith.muli %v_64, %v_58 : index
          %v_66 = arith.addi %v_57, %v_65 : index
          %v_67 = memref.load %v_24[%v_62] : memref<?xf64>
          %v_68 = memref.load %v_12[%v_53] : memref<?xf64>
          %v_69 = memref.load %v_18[%v_66] : memref<?xf64>
          %v_70 = arith.mulf %v_68, %v_69 : f64
          %v_71 = arith.addf %v_67, %v_70 : f64
          memref.store %v_71, %v_24[%v_62] : memref<?xf64>
        }
      }
    }
    %v_72 = memref.dim %v_30, %v_35 : memref<?xf64>
    scf.for %v_73 = %v_35 to %v_72 step %v_37 {
      %v_74 = arith.constant 0.0 : f64
      memref.store %v_74, %v_30[%v_73] : memref<?xf64>
    }
    scf.for %v_75 = %v_35 to %v_26 step %v_37 {
      %v_76 = llvm.extractvalue %_A_9[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_77 = arith.index_cast %v_76 : i64 to index
      %v_78 = arith.muli %v_77, %v_75 : index
      %v_79 = arith.addi %v_35, %v_78 : index
      %v_80 = llvm.extractvalue %_A_19[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_81 = arith.index_cast %v_80 : i64 to index
      %v_82 = arith.muli %v_81, %v_75 : index
      %v_83 = arith.addi %v_35, %v_82 : index
      %v_84 = llvm.extractvalue %__A_59_72[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_85 = arith.index_cast %v_84 : i64 to index
      %v_86 = arith.muli %v_85, %v_75 : index
      %v_87 = arith.addi %v_35, %v_86 : index
      %v_88 = memref.load %v_4[%v_83] : memref<?xindex>
      %v_89 = arith.addi %v_83, %v_37 : index
      %v_90 = memref.load %v_4[%v_89] : memref<?xindex>
      %v_91 = arith.cmpi slt, %v_88, %v_90 : index
      %v_95, %v_96 = scf.if %v_91 -> (index, index) {
        %v_92 = memref.load %v_6[%v_88] : memref<?xindex>
        %v_93 = arith.subi %v_90, %v_37 : index
        %v_94 = memref.load %v_6[%v_93] : memref<?xindex>
        scf.yield %v_94, %v_92 : index, index
      } else {

        scf.yield %v_35, %v_37 : index, index
      }
      %v_97 = memref.load %v_6[%v_88] : memref<?xindex>
      %v_98 = arith.cmpi slt, %v_97, %v_35 : index
      %v_101 = scf.if %v_98 -> (index) {
        %v_99 = arith.subi %v_90, %v_37 : index
        %v_100 = func.call @scansearch(%v_6, %v_35, %v_88, %v_99) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_100 : index
      } else {
        scf.yield %v_88 : index
      }
      %v_134:3 = scf.while (%v_102 = %v_35, %v_103 = %v_101, %v_104 = %v_96) : (index, index, index) -> (index, index, index) {
        %v_105 = arith.addi %v_95, %v_37 : index
        %v_106 = arith.minsi %v_28, %v_105 : index
        %v_107 = arith.cmpi slt, %v_104, %v_106 : index
        scf.condition(%v_107) %v_102, %v_103, %v_104 : index, index, index
      } do {
        ^bb(%v_102: index, %v_103: index, %v_104: index):
        %v_108 = arith.addi %v_104, %v_37 : index
        %v_109 = arith.minsi %v_108, %v_104 : index
        scf.for %v_110 = %v_102 to %v_109 step %v_37 {
          %v_111 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_112 = arith.index_cast %v_111 : i64 to index
          %v_113 = arith.muli %v_112, %v_110 : index
          %v_114 = arith.addi %v_79, %v_113 : index
        }
        %v_115 = arith.maxsi %v_102, %v_104 : index
        %v_116 = arith.addi %v_104, %v_37 : index
        scf.for %v_117 = %v_115 to %v_116 step %v_37 {
          %v_118 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_119 = arith.index_cast %v_118 : i64 to index
          %v_120 = arith.muli %v_119, %v_117 : index
          %v_121 = arith.addi %v_79, %v_120 : index
          %v_122 = llvm.extractvalue %__A_59_72[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_123 = arith.index_cast %v_122 : i64 to index
          %v_124 = arith.muli %v_123, %v_117 : index
          %v_125 = arith.addi %v_87, %v_124 : index
          %v_126 = memref.load %v_2[%v_103] : memref<?xf64>
          %v_127 = memref.load %v_24[%v_125] : memref<?xf64>
          %v_128 = arith.mulf %v_126, %v_127 : f64
          memref.store %v_128, %v_30[%v_121] : memref<?xf64>
        }
        %v_129 = arith.addi %v_104, %v_37 : index
        %v_130 = arith.addi %v_103, %v_37 : index
        %v_131 = arith.cmpi slt, %v_130, %v_90 : index
        %v_133 = scf.if %v_131 -> (index) {
          %v_132 = memref.load %v_6[%v_130] : memref<?xindex>
          scf.yield %v_132 : index
        } else {

          scf.yield %v_10 : index
        }
        scf.yield %v_129, %v_130, %v_133 : index, index, index
      }
      %v_135 = arith.addi %v_95, %v_37 : index
      %v_136 = arith.maxsi %v_35, %v_135 : index
      scf.for %v_137 = %v_136 to %v_28 step %v_37 {
        %v_138 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_139 = arith.index_cast %v_138 : i64 to index
        %v_140 = arith.muli %v_139, %v_137 : index
        %v_141 = arith.addi %v_79, %v_140 : index
      }
    }
    %v_142 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_143 = llvm.insertvalue %_A_9, %v_142[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_143, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
