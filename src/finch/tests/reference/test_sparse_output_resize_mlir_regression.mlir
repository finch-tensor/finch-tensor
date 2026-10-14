module {
  func.func @sparse_output(%idx: !llvm.ptr, %val: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.getelementptr %idx[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_2 = llvm.load %v : !llvm.ptr -> !llvm.ptr
    %v_3 = llvm.getelementptr %idx[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_4 = llvm.load %v_3 : !llvm.ptr -> i64
    %v_5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_6 = arith.constant 0 : i64
    %v_7 = arith.constant 1 : i64
    %v_8 = llvm.insertvalue %v_2, %v_5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_9 = llvm.insertvalue %v_2, %v_8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_10 = llvm.insertvalue %v_6, %v_9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_11 = llvm.insertvalue %v_4, %v_10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_12 = llvm.insertvalue %v_7, %v_11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = builtin.unrealized_conversion_cast %v_12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_14 = llvm.getelementptr %val[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_15 = llvm.load %v_14 : !llvm.ptr -> !llvm.ptr
    %v_16 = llvm.getelementptr %val[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_17 = llvm.load %v_16 : !llvm.ptr -> i64
    %v_18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_19 = llvm.insertvalue %v_15, %v_18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_20 = llvm.insertvalue %v_15, %v_19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_21 = llvm.insertvalue %v_6, %v_20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_22 = llvm.insertvalue %v_17, %v_21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_23 = llvm.insertvalue %v_7, %v_22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_24 = builtin.unrealized_conversion_cast %v_23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_25 = arith.constant 1 : index
    %v_26 = arith.index_cast %v_25 : index to i64
    %v_27 = llvm.getelementptr %idx[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_28 = llvm.getelementptr %idx[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_29 = llvm.load %v_28 : !llvm.ptr -> !llvm.ptr
    %v_30 = llvm.call %v_29(%v_27, %v_26) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_32 = llvm.insertvalue %v_30, %v_31[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_33 = llvm.insertvalue %v_30, %v_32[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_34 = llvm.insertvalue %v_6, %v_33[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_35 = llvm.insertvalue %v_26, %v_34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_36 = llvm.insertvalue %v_7, %v_35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = builtin.unrealized_conversion_cast %v_36 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_38 = arith.index_cast %v_25 : index to i64
    %v_39 = llvm.getelementptr %val[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_40 = llvm.getelementptr %val[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_41 = llvm.load %v_40 : !llvm.ptr -> !llvm.ptr
    %v_42 = llvm.call %v_41(%v_39, %v_38) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.insertvalue %v_42, %v_43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_45 = llvm.insertvalue %v_42, %v_44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_46 = llvm.insertvalue %v_6, %v_45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_47 = llvm.insertvalue %v_38, %v_46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_48 = llvm.insertvalue %v_7, %v_47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_49 = builtin.unrealized_conversion_cast %v_48 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_50 = arith.constant 2 : index
    %v_51 = arith.constant 0 : index
    memref.store %v_50, %v_37[%v_51] : memref<?xindex>
    %v_52 = arith.constant 3.5 : f64
    memref.store %v_52, %v_49[%v_51] : memref<?xf64>
    %v_53 = memref.dim %v_37, %v_51 : memref<?xindex>
    %v_54 = arith.index_cast %v_53 : index to i64
    %v_55 = llvm.getelementptr %idx[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_54, %v_55 : i64, !llvm.ptr
    %v_56 = memref.extract_aligned_pointer_as_index %v_37 : memref<?xindex> -> index
    %v_57 = arith.index_cast %v_56 : index to i64
    %v_58 = llvm.inttoptr %v_57 : i64 to !llvm.ptr
    %v_59 = llvm.getelementptr %idx[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_58, %v_59 : !llvm.ptr, !llvm.ptr
    %v_60 = memref.dim %v_49, %v_51 : memref<?xf64>
    %v_61 = arith.index_cast %v_60 : index to i64
    %v_62 = llvm.getelementptr %val[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_61, %v_62 : i64, !llvm.ptr
    %v_63 = memref.extract_aligned_pointer_as_index %v_49 : memref<?xf64> -> index
    %v_64 = arith.index_cast %v_63 : index to i64
    %v_65 = llvm.inttoptr %v_64 : i64 to !llvm.ptr
    %v_66 = llvm.getelementptr %val[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_65, %v_66 : !llvm.ptr, !llvm.ptr
    func.return
  }
}
